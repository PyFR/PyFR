import numpy as np

from pyfr.integrators.dual.pseudo.base import BaseDualPseudoIntegrator
from pyfr.mpiutil import get_comm_rank_root, mpi


class BaseDualPseudoController(BaseDualPseudoIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure the system is compatible with our formulation
        self.system.elementscls.validate_formulation(self)

        # Stats on the most recent step
        self.pseudostepinfo = []

    def convmon(self, i, minniters, dt_fac=1):
        if i >= minniters - 1:
            # Compute the normalised residual
            resid = self._resid(self._idxcurr, self._idxprev, dt_fac)

            self._update_pseudostepinfo(i + 1, resid)
            return all(r <= t for r, t in zip(resid, self._pseudo_residtol))
        else:
            self._update_pseudostepinfo(i + 1, None)
            return False

    def _resid(self, rcurr, rold, dt_fac):
        comm, rank, root = get_comm_rank_root()

        # Get a set of kernels to compute the residual
        rkerns = self._get_reduction_kerns(rcurr, rold, method='resid',
                                           norm=self._pseudo_norm)

        # Bind the dynmaic arguments
        for kern in rkerns:
            kern.bind(dt_fac)

        # Run the kernels
        self.backend.run_kernels(rkerns, wait=True)

        # Pseudo L2 norm
        if self._pseudo_norm == 'l2':
            # Reduce locally (element types) and globally (MPI ranks)
            res = np.array([sum(e) for e in zip(*[r.retval for r in rkerns])])
            comm.Allreduce(mpi.IN_PLACE, res, op=mpi.SUM)

            # Normalise and return
            return tuple(np.sqrt(res / self._gndofs))
        # Uniform norm
        else:
            # Reduce locally (element types) and globally (MPI ranks)
            res = np.array([max(e) for e in zip(*[r.retval for r in rkerns])])
            comm.Allreduce(mpi.IN_PLACE, res, op=mpi.MAX)

            # Normalise and return
            return tuple(np.sqrt(res))

    def _update_pseudostepinfo(self, niters, resid):
        self.pseudostepinfo.append((self.ntotiters, niters, resid))

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, y):
        if self.cfg.get('solver-time-integrator', 'pseudo-controller') == 'local-pi':        
            self.dtau_multiplied(y/self._dt)
        self._dt = y

class DualNonePseudoController(BaseDualPseudoController):
    pseudo_controller_name = 'none'
    pseudo_controller_needs_lerrest = False

    def pseudo_advance(self, tcurr):
        self.tcurr = tcurr

        for i in range(self.maxniters):
            # Take the step
            self._idxcurr, self._idxprev = self.step(self.tcurr)

            # Convergence monitoring
            if self.convmon(i, self.minniters, self._dtau):
                break


class DualPIPseudoController(BaseDualPseudoController):
    pseudo_controller_name = 'local-pi'
    pseudo_controller_needs_lerrest = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sect = 'solver-time-integrator'

        # Error norm
        self._norm = self.cfg.get(sect, 'errest-norm', 'l2')
        if self._norm not in {'l2', 'uniform'}:
            raise ValueError('Invalid error norm')

        tplargs = {'nvars': self.system.nvars}

        # Error tolerance
        tplargs['atol'] = self.cfg.getfloat(sect, 'atol')

        # PI control values
        sord = self.pseudo_stepper_order
        tplargs['expa'] = self.cfg.getfloat(sect, 'pi-alpha', 0.7) / sord
        tplargs['expb'] = self.cfg.getfloat(sect, 'pi-beta', 0.4) / sord

        # Constants
        tplargs['maxf'] = self.cfg.getfloat(sect, 'max-fact', 1.01)
        tplargs['minf'] = self.cfg.getfloat(sect, 'min-fact', 0.98)
        tplargs['saff'] = self.cfg.getfloat(sect, 'safety-fact', 0.8)
        
        dtau_minf = self.cfg.getfloat(sect, 'pseudo-dt-min-mult', 1.0)
        dtau_maxf = self.cfg.getfloat(sect, 'pseudo-dt-max-mult', 3.0)

        self._dtau_min = dtau_minf * self._dtau
        self._dtau_max = dtau_maxf * self._dtau
        self.dtau_fieldf = 1.0

        if not tplargs['minf'] < 1 <= tplargs['maxf']:
            raise ValueError('Invalid pseudo max-fact, min-fact')

        if not dtau_minf <= 1 < dtau_maxf:
            raise ValueError('Invalid pseudo-dt-max-mult, pseudo-dt-min-mult')

        # Register a kernel to compute local error
        self.backend.pointwise.register(
            'pyfr.integrators.dual.pseudo.kernels.localerrest'
        )

        # Storage for kernels that require dtau-min and dtau-max at runtime
        self.ele_scal_upts_locs = []
        
        for ele, shape, dtaumat in zip(self.system.ele_map.values(),
                                       self.system.ele_shapes, self.dtau_upts):
            # Allocate storage for previous error
            err_prev = self.backend.matrix(shape, np.ones(shape),
                                           tags={'align'})

            # Append the error kernels to the list
            for i, err in enumerate(ele.scal_upts):
                self.ele_scal_upts_locs.append(i)
                self.pintgkernels['localerrest', i].append(
                    self.backend.kernel(
                        'localerrest', tplargs=tplargs,
                        dims=[ele.nupts, ele.neles], err=err,
                        errprev=err_prev, dtau_upts=dtaumat
                    )
                )

            for i in self.ele_scal_upts_locs:
                for k in self.pintgkernels['localerrest', i]:
                    k.bind(dtau_min = self._dtau_min, dtau_max = self._dtau_max,
                           dtau_fieldf = self.dtau_fieldf)

        self.backend.commit()

    def dtau_multiplied(self, y):
        self.dtau_fieldf = y
        self._dtau_min *= y
        self._dtau_max *= y

        for i in self.ele_scal_upts_locs:
            for k in self.pintgkernels['localerrest', i]:
                k.bind(dtau_min = self._dtau_min, dtau_max = self._dtau_max,
                       dtau_fieldf = self.dtau_fieldf)

    def dtau_multiplier_reset(self):
        self.dtau_fieldf = 1.0
        for i in self.ele_scal_upts_locs:
            for k in self.pintgkernels['localerrest', i]:
                k.bind(dtau_min = self._dtau_min, dtau_max = self._dtau_max,
                       dtau_fieldf = 1.0)

    def localerrest(self, errbank):
        self.backend.run_kernels(self.pintgkernels['localerrest', errbank])

    def pseudo_advance(self, tcurr):
        self.tcurr = tcurr

        for i in range(self.maxniters):
            # Take the step
            self._idxcurr, self._idxprev, self._idxerr = self.step(self.tcurr)
            self.localerrest(self._idxerr)

            if i == 0 and self.dtau_fieldf != 1.0:
                self.dtau_multiplier_reset()

            if self.convmon(i, self.minniters):
                break
