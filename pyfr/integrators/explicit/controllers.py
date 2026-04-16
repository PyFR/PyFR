from pyfr.integrators.base import StepInfo
from pyfr.integrators.controllers import CFLControllerMixin, PIControllerMixin
from pyfr.integrators.explicit.base import BaseExplicitIntegrator


class BaseExplicitController(BaseExplicitIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Stats on the most recent step
        self.stepinfo = []

        # Fire off any event handlers if not restarting
        if not self.isrestart:
            self._run_plugins()

    def _accept_step(self, dt, idxcurr, wtime, err=None):
        self._advance_time(dt)
        self.nacptsteps += 1
        self.stepinfo.append(StepInfo(dt, 'accept', err, wtime))

        self.idxcurr = idxcurr

        self._invalidate_caches()

        # Run any plugins
        self._run_plugins()

        # Clear the step info
        self.stepinfo = []

    def _reject_step(self, dt, idxold, wtime, err=None):
        if dt <= self.dtmin:
            raise RuntimeError('Minimum sized time step rejected')

        self.nrjctsteps += 1
        self.stepinfo.append(StepInfo(dt, 'reject', err, wtime))

        self.idxcurr = idxold


class ExplicitNoneController(BaseExplicitController):
    controller_name = 'none'
    controller_has_variable_dt = False
    controller_needs_errest = False
    controller_needs_cfl = False

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            # Decide on the time step
            dt = self._clamp_dt(self.dt, t)

            # Take the step
            idxcurr, wtime = self._timed_step(self.tcurr, dt)

            # We are not adaptive, so accept every step
            self._accept_step(dt, idxcurr, wtime)


class ExplicitCFLController(CFLControllerMixin, BaseExplicitController):
    pass


class ExplicitPIController(PIControllerMixin, BaseExplicitController):
    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        self._init_pi_controller()

        f = self.cfg.getfloat

        self._pi_alpha = f('solver-time-integrator', 'pi-alpha', 0.58)
        self._pi_beta = f('solver-time-integrator', 'pi-beta', 0.42)

        # Initialise dt/errprev from restart or defaults
        if initsoln and (sd := initsoln.state.get('intg/ctrl')) is not None:
            self.dt, self._errprev = sd
            diff = self.cfg.sect_diff(initsoln.config, 'solver-time-integrator')
            if any(k.startswith(('atol', 'rtol')) for k in diff):
                self._errprev = 1.0
        else:
            self._errprev = 1.0

        self.serialiser.register('intg/ctrl',
                                 lambda: [self.dt, self._errprev])

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        # Constants
        maxf = self._maxfac
        minf = self._minfac
        saff = self._saffac
        sord = self.stepper_order

        expa = self._pi_alpha / sord
        expb = self._pi_beta / sord

        while self.tcurr < t:
            # Decide on the time step
            dt = self._clamp_dt(min(self.dt, self.dtmax), t)

            # Take the step
            (icurr, iprev, ierr), wtime = self._timed_step(self.tcurr, dt)

            # Estimate the error
            err = self._errest(icurr, ierr)

            # Determine time step adjustment factor
            fac = err**-expa * self._errprev**expb
            fac = min(maxf, max(minf, saff*fac))

            # Compute the size of the next step
            self.dt = fac*dt

            # Decide if to accept or reject the step
            if err < 1.0:
                self._errprev = err
                self._accept_step(dt, icurr, wtime, err=err)
            else:
                self._reject_step(dt, iprev, wtime, err=err)
