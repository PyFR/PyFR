# -*- coding: utf-8 -*-

from abc import abstractmethod

from pyfr.integrators.base import BaseIntegrator
from pyfr.util import proxylist


class BaseStepper(BaseIntegrator):
    def __init__(self, *args, **kwargs):
        super(BaseStepper, self).__init__(*args, **kwargs)

        backend = self._backend
        elemats = self._system.ele_banks

        # Create a proxylist of matrix-banks for each storage register
        self._regs = regs = []
        self._regidx = regidx = []
        for i in xrange(self._stepper_nregs):
            b = proxylist([backend.matrix_bank(em, i) for em in elemats])
            regs.append(b)
            regidx.append(i)

        # Add kernel cache
        self._axnpby_kerns = {}

    def collect_stats(self, stats):
        super(BaseStepper, self).collect_stats(stats)

        stats.set('solver-time-integrator', 'nsteps', self.nsteps)
        stats.set('solver-time-integrator', 'nfevals', self._stepper_nfevals)

    def _get_axnpby_kerns(self, n):
        try:
            return self._axnpby_kerns[n]
        except KeyError:
            k = self._kernel('axnpby', nargs=n)

            # Cache and return
            self._axnpby_kerns[n] = k
            return k

    def _add(self, *args):
        # Get a suitable set of axnpby kernels
        axnpby = self._get_axnpby_kerns(len(args) // 2)

        # Bank indices are in odd-numbered arguments
        self._prepare_reg_banks(*args[1::2])

        # Bind and run the axnpby kernels
        self._queue % axnpby(*args[::2])


class EulerStepper(BaseStepper):
    stepper_name = 'euler'

    @property
    def _stepper_has_errest(self):
        return False

    @property
    def _stepper_nfevals(self):
        return self.nsteps

    @property
    def _stepper_nregs(self):
        return 2

    @property
    def _stepper_order(self):
        return 1

    def step(self, t, dt):
        add, rhs = self._add, self._system.rhs
        ut, f = self._regidx

        rhs(ut, f)
        add(1.0, ut, dt, f)

        return ut


class RK4Stepper(BaseStepper):
    stepper_name = 'rk4'

    @property
    def _stepper_has_errest(self):
        return False

    @property
    def _stepper_nfevals(self):
        return 4*self.nsteps

    @property
    def _stepper_nregs(self):
        return 3

    @property
    def _stepper_order(self):
        return 4

    def step(self, t, dt):
        add, rhs = self._add, self._system.rhs

        # Get the bank indices for each register
        r0, r1, r2 = self._regidx

        # Ensure r0 references the bank containing u(t)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage; r1 = -∇·f(r0)
        rhs(r0, r1)

        # Second stage; r2 = r0 + dt/2*r1; r2 = -∇·f(r2)
        add(0.0, r2, 1.0, r0, dt/2.0, r1)
        rhs(r2, r2)

        # As no subsequent stages depend on the first stage we can
        # reuse its register to start accumulating the solution with
        # r1 = r0 + dt/6*r1 + dt/3*r2
        add(dt/6.0, r1, 1.0, r0, dt/3.0, r2)

        # Third stage; here we reuse the r2 register
        # r2 = r0 + dt/2*r2
        # r2 = -∇·f(r2)
        add(dt/2.0, r2, 1.0, r0)
        rhs(r2, r2)

        # Accumulate; r1 = r1 + dt/3*r2
        add(1.0, r1, dt/3.0, r2)

        # Fourth stage; again we reuse r2
        # r2 = r0 + dt*r2
        # r2 = -∇·f(r2)
        add(dt, r2, 1.0, r0)
        rhs(r2, r2)

        # Final accumulation r1 = r1 + dt/6*r2 = u(t + dt)
        add(1.0, r1, dt/6.0, r2)

        # Return the index of the bank containing u(t + dt)
        return r1


class RK45Stepper(BaseStepper):
    stepper_name = 'rk45'

    @property
    def _stepper_has_errest(self):
        return False

    @property
    def _stepper_nfevals(self):
        return 5*self.nsteps

    @property
    def _stepper_nregs(self):
        return 2

    @property
    def _stepper_order(self):
        return 4

    def step(self, t, dt):
        a21 = 970286171893 / 4311952581923.0
        a32 = 6584761158862 / 12103376702013.0
        a43 = 2251764453980 / 15575788980749.0
        a54 = 26877169314380 / 34165994151039.0

        b1 = 1153189308089 / 22510343858157.0
        b2 = 1772645290293 / 4653164025191.0
        b3 = -1672844663538 / 4480602732383.0
        b4 = 2114624349019 / 3568978502595.0
        b5 = 5198255086312 / 14908931495163.0

        add, rhs = self._add, self._system.rhs
        r1, r2 = self._regidx

        rhs(r1, r2)
        add(1.0, r1, a21*dt, r2)
        add((b1 - a21)*dt, r2, 1.0, r1)

        rhs(r1, r1)
        add(1.0, r2, a32*dt, r1)
        add((b2 - a32)*dt, r1, 1.0, r2)

        rhs(r2, r2)
        add(1.0, r1, a43*dt, r2)
        add((b3 - a43)*dt, r2, 1.0, r1)

        rhs(r1, r1)
        add(1.0, r2, a54*dt, r1)
        add((b4 - a54)*dt, r1, 1.0, r2)

        rhs(r2, r2)
        add(1.0, r1, b5*dt, r2)

        return r1
