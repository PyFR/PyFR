from pyfr.cache import memoize
from pyfr.integrators.explicit.base import BaseExplicitIntegrator
from pyfr.integrators.registers import (DynamicScalarRegister,
                                        DynamicVectorRegister,
                                        ScalarRegister, VectorRegister)


class BaseExplicitStepper(BaseExplicitIntegrator):
    pass


class ExplicitEulerStepper(BaseExplicitStepper):
    stepper_name = 'euler'
    stepper_has_errest = False
    stepper_order = 1

    _ut = ScalarRegister()
    _f = ScalarRegister()

    def step(self, t, dt):
        self._rhs(t, self._ut, self._f)
        self._add(1.0, self._ut, dt, self._f, comp=self._ut)

        return self._ut


class TVDRK3Stepper(BaseExplicitStepper):
    stepper_name = 'tvd-rk3'
    stepper_has_errest = False
    stepper_order = 3

    # Solution and stage registers plus a second flux when compensated
    _regidx = DynamicVectorRegister()

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        # Make the config available before the registers are sized
        self.cfg = cfg

        # Size the registers before they are assigned
        self._size_register(self._regidx, 4 if self.comp_accum else 3)

        super().__init__(backend, systemcls, mesh, initsoln, cfg)

    def step(self, t, dt):
        add, rhs = self._add, self._rhs

        if self.comp_accum:
            # Get the bank indices for each register (n, stage, f0, f1)
            r0, r1, r2, r3 = self._regidx

            # First stage; r2 = -∇·f(r0); r1 = r0 + dt*r2
            rhs(t, r0, r2)
            add(0.0, r1, 1.0, r0, dt, r2, comp=r0)

            # Second stage; r3 = -∇·f(r1); r1 = r0 + 0.25*dt*(r2 + r3)
            rhs(t + dt, r1, r3)
            add(0.0, r1, 1.0, r0, 0.25*dt, r2, 0.25*dt, r3, comp=r0)

            # Third stage; r1 = -∇·f(r1); r0 += dt/6*(r2 + r3) + 2/3*dt*r1
            rhs(t + 0.5*dt, r1, r1)
            add(1.0, r0, dt/6.0, r2, dt/6.0, r3, 2.0/3.0*dt, r1, comp=r0)

            # Return the index of the bank containing u(t + dt)
            return r0
        else:
            # Get the bank indices for each register (n, n+1, rhs)
            r0, r1, r2 = self._regidx

            # Ensure r0 references the bank containing u(t)
            if r0 != self.idxcurr:
                r0, r1 = r1, r0

            # First stage; r2 = -∇·f(r0); r1 = r0 + dt*r2
            rhs(t, r0, r2)
            add(0.0, r1, 1.0, r0, dt, r2)

            # Second stage; r2 = -∇·f(r1); r1 = 0.75*r0 + 0.25*r1 + 0.25*dt*r2
            rhs(t + dt, r1, r2)
            add(0.25, r1, 0.75, r0, 0.25*dt, r2)

            # Third stage; r2 = -∇·f(r1); r1 = r0/3 + 2/3*r1 + 2/3*dt*r2
            rhs(t + 0.5*dt, r1, r2)
            add(2.0/3.0, r1, 1.0/3.0, r0, 2.0/3.0*dt, r2)

            # Return the index of the bank containing u(t + dt)
            return r1


class RK4Stepper(BaseExplicitStepper):
    stepper_name = 'rk4'
    stepper_has_errest = False
    stepper_order = 4

    _regidx = VectorRegister(n=3)

    def step(self, t, dt):
        add, rhs = self._add, self._rhs
        comp_accum = self.comp_accum

        # Get the bank indices for each register
        r0, r1, r2 = self._regidx

        # Ensure r0 references the bank containing u(t)
        if r0 != self.idxcurr:
            r0, r1 = r1, r0

        # First stage; r1 = -∇·f(r0)
        rhs(t, r0, r1)

        # Second stage; r2 = r0 + dt/2*r1; r2 = -∇·f(r2)
        add(0.0, r2, 1.0, r0, dt/2.0, r1, comp=r0)
        rhs(t + dt/2.0, r2, r2)

        # Reuse the spent first stage register to accumulate the step
        if comp_accum:
            # Keep the solution out of the increment; r1 = dt/6*r1 + dt/3*r2
            add(dt/6.0, r1, dt/3.0, r2)
        else:
            # r1 = r0 + dt/6*r1 + dt/3*r2
            add(dt/6.0, r1, 1.0, r0, dt/3.0, r2)

        # Third stage; here we reuse the r2 register
        # r2 = r0 + dt/2*r2
        # r2 = -∇·f(r2)
        add(dt/2.0, r2, 1.0, r0, comp=r0)
        rhs(t + dt/2.0, r2, r2)

        # Accumulate; r1 = r1 + dt/3*r2
        add(1.0, r1, dt/3.0, r2)

        # Fourth stage; again we reuse r2
        # r2 = r0 + dt*r2
        # r2 = -∇·f(r2)
        add(dt, r2, 1.0, r0, comp=r0)
        rhs(t + dt, r2, r2)

        # Final accumulation into the compensated solution or the increment
        if comp_accum:
            # r0 = r0 + r1 + dt/6*r2 = u(t + dt)
            add(1.0, r0, 1.0, r1, dt/6.0, r2, comp=r0)

            return r0
        else:
            # r1 = r1 + dt/6*r2 = u(t + dt)
            add(1.0, r1, dt/6.0, r2)

            return r1


class RKVdH2RStepper(BaseExplicitStepper):
    # Solution, stage, and, if not compensated, old solution registers
    _regidx = DynamicVectorRegister()

    # Step increment for compensated stepping
    _rinc = DynamicScalarRegister()

    # Error estimate
    _rerr = DynamicScalarRegister()

    # Coefficients
    a = []
    b = []
    bhat = []

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        # Make the config available before the registers are sized
        self.cfg = cfg

        # Size the registers before they are assigned
        comp_accum, errest = self.comp_accum, self.stepper_has_errest
        nregs = 3 if errest and not comp_accum else 2
        self._size_register(self._regidx, nregs)
        self._size_register(self._rinc, int(comp_accum))
        self._size_register(self._rerr, int(errest))

        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        # Register our pointwise kernel
        self.backend.pointwise.register(
            'pyfr.integrators.explicit.kernels.rkvdh2'
        )

        # Compute the coefficients
        self.c = [0.0] + [sum(self.b[:i]) + ai for i, ai in enumerate(self.a)]
        self.e = [b - bh for b, bh in zip(self.b, self.bhat)]

        self._nstages = len(self.c)

    @memoize
    def _get_rkvdh2_kerns(self, stage, r1, r2, rold=None):
        kerns = []
        errest, comp_accum = self.stepper_has_errest, self.comp_accum
        tplargs = {
            'a': self.a, 'b': self.b, 'e': self.e,
            'stage': stage, 'nstages': self._nstages,
            'nvars': self.system.nvars, 'errest': errest,
            'comp_accum': comp_accum
        }

        # Renormalise into the solution bank unless the step may be rejected
        rout = r2 if errest else r1

        shapes, banks = self.system.ele_shapes.values(), self.system.ele_banks
        for (nupts, _, neles), em, cm in zip(shapes, banks, self._comp):
            kargs = {'r1': em[r1], 'r2': em[r2]}

            if errest:
                kargs['rerr'] = em[self._rerr]

            if rold is not None:
                kargs['rold'] = em[rold]

            if comp_accum:
                kargs |= {'rinc': em[self._rinc], 'r1c': cm[r1],
                          'routc': cm[rout]}

            kern = self.backend.kernel('rkvdh2', tplargs=tplargs,
                                       dims=[nupts, neles], **kargs)
            kerns.append(kern)

        return kerns

    @property
    def stepper_has_errest(self):
        return self.controller_needs_errest and bool(self.bhat)

    def _comp_accum_banks(self):
        # Compensate both solution banks when steps may be rejected
        if self.stepper_has_errest:
            return self._regidx
        else:
            return super()._comp_accum_banks()

    def step(self, t, dt):
        run_kernels = self.backend.run_kernels

        # Solution bank followed by the stage bank and any old solution bank
        r1 = self.idxcurr
        r2, *rold = (r for r in self._regidx if r != r1)

        # Evaluate the stages in the scheme
        for i, ci in enumerate(self.c):
            # Compute -∇·f
            self._rhs(t + ci*dt, r2 if i > 0 else r1, r2)

            # Fetch the appropriate RK accumulation kernels
            kerns = self._get_rkvdh2_kerns(i, r1, r2, *rold)

            # Bind the arguments
            for k in kerns:
                k.bind(dt=dt)

            # Execute
            run_kernels(kerns)

            # Swap unless the solution is held fixed for compensation
            if not self.comp_accum:
                r1, r2 = r2, r1

        # Return the bank containing u(t + dt)
        if not self.stepper_has_errest:
            return r1 if self.comp_accum else r2
        # Return u(t + dt), the retained u(t), and the error estimate
        elif self.comp_accum:
            return (r2, r1, self._rerr)
        # Return u(t + dt), the copy of u(t), and the error estimate
        else:
            return (r2, *rold, self._rerr)


class RK34Stepper(RKVdH2RStepper):
    stepper_name = 'rk34'
    stepper_order = 3

    a = [
        11847461282814 / 36547543011857,
        3943225443063 / 7078155732230,
        -346793006927 / 4029903576067
    ]

    b = [
        1017324711453 / 9774461848756,
        8237718856693 / 13685301971492,
        57731312506979 / 19404895981398,
        -101169746363290 / 37734290219643
    ]

    bhat = [
        15763415370699 / 46270243929542,
        514528521746 / 5659431552419,
        27030193851939 / 9429696342944,
        -69544964788955 / 30262026368149
    ]


class RK45Stepper(RKVdH2RStepper):
    stepper_name = 'rk45'
    stepper_order = 4

    a = [
        970286171893 / 4311952581923,
        6584761158862 / 12103376702013,
        2251764453980 / 15575788980749,
        26877169314380 / 34165994151039
    ]

    b = [
        1153189308089 / 22510343858157,
        1772645290293 / 4653164025191,
        -1672844663538 / 4480602732383,
        2114624349019 / 3568978502595,
        5198255086312 / 14908931495163
    ]

    bhat = [
        1016888040809 / 7410784769900,
        11231460423587 / 58533540763752,
        -1563879915014 / 6823010717585,
        606302364029 / 971179775848,
        1097981568119 / 3980877426909
    ]
