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
            resid, if_converged = self._resid_multiple(self._idxcurr, self._idxprev, dt_fac)

            self._update_pseudostepinfo(i + 1, resid)
            return if_converged
        else:
            self._update_pseudostepinfo(i + 1, None)
            return False

    def _resid_multiple(self, rcurr, rold, dt_fac):

        if self._pseudo_norm in ('l2', 'l4', 'l8', 'uniform'):
            resid = self._resid(rcurr, rold, dt_fac, self._pseudo_norm)                
            return resid, all(r <= t for r, t in zip(resid, self._pseudo_residtol))

        elif self._pseudo_norm == 'all':

            resid_l2 = self._resid(rcurr, rold, dt_fac, 'l2')
            resid_l4 = self._resid(rcurr, rold, dt_fac, 'l4')
            resid_l8 = self._resid(rcurr, rold, dt_fac, 'l8')
            resid_li = self._resid(rcurr, rold, dt_fac, 'uniform')

            if_converged_l2 = all(r <= t for r, t in zip(resid_l2, self._pseudo_residtol_l2))
            if_converged_l4 = all(r <= t for r, t in zip(resid_l4, self._pseudo_residtol_l4))
            if_converged_l8 = all(r <= t for r, t in zip(resid_l8, self._pseudo_residtol_l8))
            if_converged_li = all(r <= t for r, t in zip(resid_li, self._pseudo_residtol_li))

            if_converged = if_converged_l2 and if_converged_l4 and if_converged_l8 and if_converged_li

            return (*resid_l2, *resid_l4, *resid_l8, *resid_li), if_converged

        else:
            raise ValueError('Invalid pseudo-norm entered.')

    def _resid(self, rcurr, rold, dt_fac, pseudo_norm):
        comm, rank, root = get_comm_rank_root()

        # Get a set of kernels to compute the residual
        rkerns = self._get_reduction_kerns(rcurr, rold, method='resid',
                                           norm=pseudo_norm)

        # Bind the dynmaic arguments
        for kern in rkerns:
            kern.bind(dt_fac)

        # Run the kernels
        self.backend.run_kernels(rkerns, wait=True)

        # Pseudo L2 norm
        if pseudo_norm == 'l2':
            # Reduce locally (element types) and globally (MPI ranks)
            res = np.array([sum(e) for e in zip(*[r.retval for r in rkerns])])
            comm.Allreduce(mpi.IN_PLACE, res, op=mpi.SUM)

            # Normalise and return
            return tuple(np.sqrt(res / self._gndofs))
        # Pseudo L4 norm
        elif pseudo_norm == 'l4':
            # Reduce locally (element types) and globally (MPI ranks)
            res = np.array([sum(e) for e in zip(*[r.retval for r in rkerns])])
            comm.Allreduce(mpi.IN_PLACE, res, op=mpi.SUM)

            # Normalise and return
            return tuple( np.sqrt(np.sqrt(res / self._gndofs)))
        # Pseudo L4 norm
        elif pseudo_norm == 'l8':
            # Reduce locally (element types) and globally (MPI ranks)
            res = np.array([sum(e) for e in zip(*[r.retval for r in rkerns])])
            comm.Allreduce(mpi.IN_PLACE, res, op=mpi.SUM)

            # Normalise and return
            return tuple( np.sqrt(np.sqrt(np.sqrt(res / self._gndofs))))
        # Uniform norm
        else:
            # Reduce locally (element types) and globally (MPI ranks)
            res = np.array([max(e) for e in zip(*[r.retval for r in rkerns])])
            comm.Allreduce(mpi.IN_PLACE, res, op=mpi.MAX)

            # Normalise and return
            return tuple(np.sqrt(res))

    def _update_pseudostepinfo(self, niters, resid):
        self.pseudostepinfo.append((self.ntotiters, niters, resid))


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
        if self._norm not in {'l2', 'l4', 'l8', 'uniform'}:
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

        tplargs['dtau_minf'] = self.cfg.getfloat(sect, 'pseudo-dt-min-mult',
                                                 1)
        tplargs['dtau_maxf'] = self.cfg.getfloat(sect, 'pseudo-dt-max-mult',
                                                 3.0)

        tplargs['dtau_minf_p'] = self.cfg.getfloat(sect, 'pseudo-dt-min-mult-p',
                                                 tplargs['dtau_minf'])
        tplargs['dtau_maxf_p'] = self.cfg.getfloat(sect, 'pseudo-dt-max-mult-p',
                                                 tplargs['dtau_maxf'])

        if not tplargs['minf'] < 1 <= tplargs['maxf']:
            raise ValueError('Invalid pseudo max-fact, min-fact')

        if not tplargs['dtau_minf'] < 1 < tplargs['dtau_maxf']:
            raise ValueError('Invalid pseudo-dt-min-mult, pseudo-dt-max-mult')

        # Limits for the local pseudo-time-step size
        tplargs['dtau_min'] = tplargs['dtau_minf'] * self._dtau
        tplargs['dtau_max'] = tplargs['dtau_maxf'] * self._dtau

        tplargs['dtau_min_p'] = tplargs['dtau_minf'] * self._dtau 
        tplargs['dtau_max_p'] = tplargs['dtau_maxf'] * self._dtau 

        # Register a kernel to compute local error
        self.backend.pointwise.register(
            'pyfr.integrators.dual.pseudo.kernels.localerrest'
        )

        for ele, shape, dtaumat in zip(self.system.ele_map.values(),
                                       self.system.ele_shapes, self.dtau_upts):
            # Allocate storage for previous error
            err_prev = self.backend.matrix(shape, np.ones(shape),
                                           tags={'align'})

            # Append the error kernels to the list
            for i, err in enumerate(ele.scal_upts):
                self.pintgkernels['localerrest', i].append(
                    self.backend.kernel(
                        'localerrest', tplargs=tplargs,
                        dims=[ele.nupts, ele.neles], err=err,
                        errprev=err_prev, dtau_upts=dtaumat
                    )
                )

        self.backend.commit()

    def localerrest(self, errbank):
        self.backend.run_kernels(self.pintgkernels['localerrest', errbank])

    def pseudo_advance(self, tcurr):
        self.tcurr = tcurr

        for i in range(self.maxniters):
            # Take the step
            self._idxcurr, self._idxprev, self._idxerr = self.step(self.tcurr)
            self.localerrest(self._idxerr)

            if self.convmon(i, self.minniters):
                break
