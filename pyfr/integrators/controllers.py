import math

from pyfr.integrators.base import kernel_getter
from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll


class CFLControllerMixin:
    controller_name = 'cfl'
    controller_has_variable_dt = True
    controller_needs_errest = False
    controller_needs_cfl = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sect = 'solver-time-integrator'
        self._order = self.cfg.getint('solver', 'order')
        self._cfl = self.cfg.getfloat(sect, 'cfl')
        self.dtmax = self.cfg.getfloat(sect, 'dt-max', 1e2)
        self._cfl_nsteps = self.cfg.getint(sect, 'cfl-nsteps', 1)

    def _compute_dt_cfl(self, uinbank):
        comm, rank, root = get_comm_rank_root()
        local_max = self.system.compute_max_wavespeed(uinbank)
        global_max = scal_coll(comm.Allreduce, local_max, op=mpi.MAX)
        return self._cfl / (global_max * (2*self._order + 1))

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            if self.nacptsteps % self._cfl_nsteps == 0:
                self.dt = self._compute_dt_cfl(self.idxcurr)

            dt = self._clamp_dt(min(self.dt, self.dtmax), t)

            idxcurr, wtime = self._timed_step(self.tcurr, dt)

            self._accept_step(dt, idxcurr, wtime)


class PIControllerMixin:
    controller_name = 'pi'
    controller_needs_errest = True
    controller_needs_cfl = False
    controller_has_variable_dt = True

    def _init_pi_controller(self):
        f = lambda k, d=None: self.cfg.getfloat('solver-time-integrator', k, d)
        g = lambda k, d=None: self.cfg.get('solver-time-integrator', k, d)
        h = lambda k: self.cfg.hasopt('solver-time-integrator', k)

        self.dtmax = f('dt-max', 1e2)

        self._rtol = f('rtol')
        if self._rtol < 10*self.backend.fpdtype_eps:
            raise ValueError('Relative tolerance too small')

        convars = self.system.elementscls.convars(self.system.ndims, self.cfg)
        pvar_atol_set = [h(f'atol-{v}') for v in convars]

        if any(pvar_atol_set) and not all(pvar_atol_set):
            raise ValueError('Missing atol for some variables')

        if all(pvar_atol_set):
            self._atols = tuple(f(f'atol-{v}') for v in convars)
        else:
            self._atols = (f('atol'),)*len(convars)

        if any(a < 10*self.backend.fpdtype_eps for a in self._atols):
            raise ValueError('Absolute tolerance too small')

        self._errest_norm = g('errest-norm', 'l2')
        if self._errest_norm not in {'l2', 'uniform'}:
            raise ValueError('Invalid error norm')

        self._errprev = 1.0

        self._saffac = f('safety-fact', 0.8)
        self._maxfac = f('max-fact', 1.1)
        self._minfac = f('min-fact', 0.9)

        if not self._minfac < 1 <= self._maxfac:
            raise ValueError('Invalid max-fact, min-fact')

    def _errest(self, rcurr, rerr):
        comm, rank, root = get_comm_rank_root()

        ekerns = self._get_errest_kerns(rcurr, rerr, norm=self._errest_norm)

        self.backend.run_kernels(ekerns, wait=True)

        if self._errest_norm == 'l2':
            err = sum(k.retval[0] for k in ekerns)
            err = scal_coll(comm.Allreduce, err, op=mpi.SUM)
            err = math.sqrt(err / self.gndofs)
        else:
            err = max(k.retval[0] for k in ekerns)
            err = scal_coll(comm.Allreduce, err, op=mpi.MAX)

        return err if not math.isnan(err) else 100

    @kernel_getter
    def _get_errest_kerns(self, emats, rcurr, rerr, *, norm):
        expr = 'err / (atol + rtol*fabs(curr))'

        if norm == 'uniform':
            expr, rop = f'fabs({expr})', 'max'
        else:
            expr, rop = f'({expr})*({expr})', 'sum'

        vvars = {'curr': emats[rcurr], 'err': emats[rerr]}

        kern = self.backend.kernel('reduction', rop, [expr], vvars,
                                   svars=['rtol'], pvars={'atol': self._atols})
        kern.bind(self._rtol)

        return kern
