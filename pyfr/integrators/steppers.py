# -*- coding: utf-8 -*-

from abc import abstractmethod

from pyfr.integrators.base import BaseIntegrator
from pyfr.util import proxylist

class BaseStepper(BaseIntegrator):
    def __init__(self, *args, **kwargs):
        super(BaseStepper, self).__init__(*args, **kwargs)

        # Number of steps taken
        self._nsteps = 0

        backend = self._backend
        elemats = self._meshp.ele_banks

        # Create a proxylist of matrix-banks for each storage register
        self._regs = regs = []
        self._regidx = regidx = []
        for i in xrange(self._stepper_nregs):
            b = proxylist([backend.matrix_bank(em, i) for em in elemats])
            regs.append(b)
            regidx.append(i)

        # Add kernel cache
        self._axnpby_kerns = {}

    @abstractmethod
    def step(self, t, dt):
        self._nsteps += 1

    def collect_stats(self, stats):
        super(BaseStepper, self).collect_stats(stats)

        stats.set('time-integration', 'nsteps', self._nsteps)
        stats.set('time-integration', 'nfevals', self._stepper_nfevals)

    def _get_axnpby_kerns(self, n):
        try:
            return self._axnpby_kerns[n]
        except KeyError:
            # Transpose from [nregs][neletypes] to [neletypes][nregs]
            transregs = zip(*self._regs)

            # Generate an axnpby kernel for each element type
            kerns = proxylist([])
            for tr in transregs:
                kerns.append(self._backend.kernel('axnpby', *tr[:n]))

            # Cache
            self._axnpby_kerns[n] = kerns

            return kerns

    def _add(self, *args):
        # Configure the register banks
        for reg, bidx in zip(self._regs, args[1::2]):
            reg.active = bidx

        # Get a suitable set of axnpby kernels
        axnpby = self._get_axnpby_kerns(len(args)/2)

        # Bind and run the axnpby kernels
        self._queue % axnpby(*args[::2])


class EulerStepper(BaseStepper):
    stepper_name = 'euler'

    @property
    def _stepper_has_errest(self):
        return False

    @property
    def _stepper_nfevals(self):
        return self._nsteps

    @property
    def _stepper_nregs(self):
        return 2

    def step(self, t, dt):
        super(EulerStepper, self).step(t, dt)

        add, negdivf = self._add, self._meshp
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
        return 4*self._nsteps

    @property
    def _stepper_nregs(self):
        return 5

    def step(self, t, dt):
        super(RK4Stepper, self).step(t, dt)

        add, negdivf = self._add, self._meshp

        # Get the bank indices for each register
        ut,  k1,  k2,  k3,  k4 = self._regidx

        # First stage
        negdivf(ut, k1)

        # Second stage (k2 = ut + dt/2*k1)
        add(0.0, k2, 1.0, ut, dt/2.0, k1)
        negdivf(k2, k2)

        # Third stage (k3 = ut + dt/2*k2)
        add(0.0, k3, 1.0, ut, dt/2.0, k2)
        negdivf(k3, k3)

        # Fourth stage (k4 = ut + dt*k3)
        add(0.0, k4, 1.0, ut, dt, k3)
        negdivf(k4, k4)

        # Compute u(t+dt) as k4 = dt/6*k4 + dt/3*k3 + dt/2*k2 + dt/6*k1 + ut
        add(dt/6.0, k4, dt/3.0, k3, dt/3.0, k2, dt/6.0, k1, 1.0, ut)

        # Swizzle k4 and u(t)
        self._regidx[0], self._regidx[4] = k4, ut

        # Return the index of the bank containing u(t + dt)
        return k4
