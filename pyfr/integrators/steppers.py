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
        axnpby = self._get_axnpby_kerns(len(args)/2)

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
        add, negdivf = self._add, self._system
        ut, f = self._regidx

        negdivf(ut, f)
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
        add, negdivf = self._add, self._system

        # Get the bank indices for each register
        r0, r1, r2 = self._regidx

        # Ensure r0 references the bank containing u(t)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage; r1 = -∇·f(r0)
        negdivf(r0, r1)

        # Second stage; r2 = r0 + dt/2*r1; r2 = -∇·f(r2)
        add(0.0, r2, 1.0, r0, dt/2.0, r1)
        negdivf(r2, r2)

        # As no subsequent stages depend on the first stage we can
        # reuse its register to start accumulating the solution with
        # r1 = r0 + dt/6*r1 + dt/3*r2
        add(dt/6.0, r1, 1.0, r0, dt/3.0, r2)

        # Third stage; here we reuse the r2 register
        # r2 = r0 + dt/2*r2
        # r2 = -∇·f(r2)
        add(dt/2.0, r2, 1.0, r0)
        negdivf(r2, r2)

        # Accumulate; r1 = r1 + dt/3*r2
        add(1.0, r1, dt/3.0, r2)

        # Fourth stage; again we reuse r2
        # r2 = r0 + dt*r2
        # r2 = -∇·f(r2)
        add(dt, r2, 1.0, r0)
        negdivf(r2, r2)

        # Final accumulation r1 = r1 + dt/6*r2 = u(t + dt)
        add(1.0, r1, dt/6.0, r2)

        # Return the index of the bank containing u(t + dt)
        return r1


class DOPRI5Stepper(BaseStepper):
    stepper_name = 'dopri5'

    @property
    def _stepper_has_errest(self):
        return False

    @property
    def _stepper_nfevals(self):
        return 6*self.nsteps + self.nrjctsteps + 1

    @property
    def _stepper_nregs(self):
        return 7

    @property
    def _stepper_order(self):
        return 5

    def step(self, t, dt):
        add, negdivf = self._add, self._system

        # Register bank indices (r0 = u(t); r1..6 = temp RK 'k' stages)
        r0, r1, r2, r3, r4, r5, r6 = self._regidx

        # Usually the first stage, -∇·f(r0 = u(t)), is available in
        # r1 (this is as the scheme is FSAL), except when the last step
        # was rejected.  In this case we compute it here.
        if not self.nacptchain:
            negdivf(r0, r1)

        # Second stage; r2 = r0 + dt/5*r1; r2 = -∇·f(r2)
        add(0.0, r2, 1.0, r0, dt/5.0, r1)
        negdivf(r2, r2)

        # Third stage; r3 = r0 + (3/40)*dt*r1 + (9/40)*dt*r2; r3 = -∇·f(r3)
        add(0.0, r3, 1.0, r0, 3.0/40.0*dt, r1, 9.0/40.0*dt, r2)
        negdivf(r3, r3)

        # Fourth stage
        # r4 = r0 + (44/45)*dt*r1 + (-56/15)*dt*r2 + (32/9)*dt*r3
        # r4 = -∇·f(r4)
        add(0.0, r4, 1.0, r0, 44.0/45.0*dt, r1, -56.0/15.0*dt, r2,
            32.0/9.0*dt, r3)
        negdivf(r4, r4)

        # Fifth stage
        # r5 = r0 + (19372/6561)*dt*r1 + (-25360/2187)*dt*r2
        #    + (64448/6561)*dt*r3 + (-212/729)*dt*r4
        # r5 = -∇·f(r5)
        add(0.0, r5, 1.0, r0, 19372.0/6561.0*dt, r1, -25360.0/2187.0*dt, r2,
            64448.0/6561.0*dt, r3, -212.0/729.0*dt, r4)
        negdivf(r5, r5)

        # Sixth stage; as neither the seventh stage nor the solution
        # coefficients depend on the second stage we are able to reuse its
        # register here
        # r2 = r0 + (9017/3168)*dt*r1 + (-355/33)*dt*r2 + (46732/5247)*dt*r3
        #    + (49/176)*dt*r4 + (-5103/18656)*dt*r5
        # r2 = -∇·f(r2)
        add(-355.0/33.0*dt, r2, 1.0, r0, 9017.0/3168.0*dt, r1,
            46732.0/5247.0*dt, r3, 49.0/176.0*dt, r4, -5103.0/18656.0*dt, r5)
        negdivf(r2, r2)

        # Seventh stage; note that r2 contains the sixth stage
        # r0 = r0 + (35/384)*dt*r1 + (500/1113)*dt*r3 + (125/192)*dt*r4
        #    + (-2187/6784)*dt*r5 + (11/84)*dt*r2
        # r6 = -∇·f(r0)
        add(1.0, r0, 35.0/384.0*dt, r1, 500.0/1113.0*dt, r3,
            125.0/192.0*dt, r4, -2187.0/6784.0*dt, r5, 11.0/84.0*dt, r2)
        negdivf(r0, r6)

        # Swizzle r1 (first stage) and r6 (seventh stage)
        self._regidx[1], self._regidx[6] = r6, r1

        # Return the index of the bank containing u(t + dt)
        return r0
