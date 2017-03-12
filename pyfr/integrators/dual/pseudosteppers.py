# -*- coding: utf-8 -*-

from pyfr.integrators.dual.base import BaseDualIntegrator


class BaseDualPseudoStepper(BaseDualIntegrator):
    def collect_stats(self, stats):
        super().collect_stats(stats)

        # Total number of RHS evaluations
        stats.set('solver-time-integrator', 'nfevals', self._stepper_nfevals)

        # Total number of pseudo-steps
        stats.set('solver-time-integrator', 'npseudosteps', self.npseudosteps)

    def _add_with_dts(self, *args, c):
        vals, regs = list(args[::2]), list(args[1::2])

        # Coefficients for the dual-time source term
        svals = [c*sc for sc in self._dual_time_source]

        # Normal addition
        axnpby = self._get_axnpby_kerns(len(vals))
        self._prepare_reg_banks(*regs)
        self._queue % axnpby(*vals)

        # Source addition
        axnpby = self._get_axnpby_kerns(len(svals) + 1, subdims=self._subdims)
        self._prepare_reg_banks(regs[0], self._idxcurr, *self._source_regidx)
        self._queue % axnpby(1, *svals)

    def finalise_step(self, currsoln):
        add = self._add
        pnreg = self._pseudo_stepper_nregs

        # Rotate the source registers to the right by one
        self._regidx[pnreg:] = (self._source_regidx[-1:] +
                                self._source_regidx[:-1])

        # Copy the current soln into the first source register
        add(0.0, self._regidx[pnreg], 1.0, currsoln)


class DualPseudoEulerStepper(BaseDualPseudoStepper):
    pseudo_stepper_name = 'euler'

    @property
    def _stepper_nfevals(self):
        return self.nsteps

    @property
    def _pseudo_stepper_nregs(self):
        return 2

    @property
    def _pseudo_stepper_order(self):
        return 1

    def step(self, t, dt, dtau):
        add, add_with_dts = self._add, self._add_with_dts
        rhs = self.system.rhs
        r0, r1 = self._stepper_regidx
        rat = dtau / dt

        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        rhs(t, r0, r1)
        add_with_dts(0, r1, 1, r0, dtau, r1, c=rat)

        return r1, r0


class DualPseudoTVDRK3Stepper(BaseDualPseudoStepper):
    pseudo_stepper_name = 'tvd-rk3'

    @property
    def _stepper_nfevals(self):
        return 3*self.nsteps

    @property
    def _pseudo_stepper_nregs(self):
        return 3

    @property
    def _pseudo_stepper_order(self):
        return 3

    def step(self, t, dt, dtau):
        add, add_with_dts = self._add, self._add_with_dts
        rhs = self.system.rhs
        rat = dtau / dt

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time
        r0, r1, r2 = self._stepper_regidx

        # Ensure r0 references the bank containing u(n+1,m)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage;
        # r2 = -∇·f(r0); r1 = r0 + dtau*r2 - dtau*dQ/dt;
        rhs(t, r0, r2)
        add_with_dts(0, r1, 1, r0, dtau, r2, c=rat)

        # Second stage;
        # r2 = -∇·f(r1); r1 = 3/4*r0 + 1/4*r1 + 1/4*dtau*r2 - dtau/4*dQ/dt
        rhs(t, r1, r2)
        add_with_dts(1/4, r1, 3/4, r0, dtau/4, r2, c=rat/4)

        # Third stage;
        # r2 = -∇·f(r1); r1 = 1/3*r0 + 2/3*r1 + 2/3*dtau*r2 - 2/3*dtau*dQ/dt
        rhs(t, r1, r2)
        add_with_dts(2/3, r1, 1/3, r0, 2*dtau/3, r2, c=2*rat/3)

        # Return the index of the bank containing u(n+1,m+1)
        return r1, r0


class DualPseudoRK4Stepper(BaseDualPseudoStepper):
    pseudo_stepper_name = 'rk4'

    @property
    def _stepper_nfevals(self):
        return 4*self.nsteps

    @property
    def _pseudo_stepper_nregs(self):
        return 3

    @property
    def _pseudo_stepper_order(self):
        return 4

    def step(self, t, dt, dtau):
        add, add_with_dts  = self._add, self._add_with_dts
        rhs = self.system.rhs
        rat = dtau / dt

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time
        r0, r1, r2 = self._stepper_regidx

        # Ensure r0 references the bank containing u(n+1,m)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage; r1 = -∇·f(r0)
        rhs(t, r0, r1)

        # Second stage; r2 = r0 + dtau/2*r1 - dtau/2*dQ/dt; r2 = -∇·f(r2)
        add_with_dts(0, r2, 1, r0, dtau/2, r1, c=rat/2)
        rhs(t, r2, r2)

        # As no subsequent stages depend on the first stage we can
        # reuse its register to start accumulating the solution with
        # r1 = r0 + dtau/6*r1 + dtau/3*r2 -(dtau/6+dtau/3)*dQ/dt
        add_with_dts(dtau/6, r1, 1, r0, dtau/3, r2, c=(1/6+1/3)*rat)

        # Third stage; here we reuse the r2 register
        # r2 = r0 + dtau/2*r2 - dtau/2*dQ/dt
        # r2 = -∇·f(r2)
        add_with_dts(dtau/2, r2, 1, r0, c=rat/2)
        rhs(t, r2, r2)

        # Accumulate; r1 = r1 + dtau/3*r2 - dtau/3*dQ/dt
        add_with_dts(1, r1, dtau/3, r2, c=rat/3)

        # Fourth stage; again we reuse r2
        # r2 = r0 + dtau*r2 - dtau*dQ/dt
        # r2 = -∇·f(r2)
        add_with_dts(dtau, r2, 1, r0, c=rat)
        rhs(t, r2, r2)

        # Final accumulation r1 = r1 + dtau/6*r2 - dtau/6*dQ/dt = u(n+1,m+1)
        add_with_dts(1, r1, dtau/6, r2, c=rat/6)

        # Return the index of the bank containing u(n+1,m+1)
        return r1, r0
