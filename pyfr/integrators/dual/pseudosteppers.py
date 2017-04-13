# -*- coding: utf-8 -*-

from pyfr.integrators.dual.base import BaseDualIntegrator


class BaseDualPseudoStepper(BaseDualIntegrator):
    def collect_stats(self, stats):
        super().collect_stats(stats)

        # Total number of RHS evaluations
        stats.set('solver-time-integrator', 'nfevals', self._stepper_nfevals)

        # Total number of pseudo-steps
        stats.set('solver-time-integrator', 'npseudosteps', self.npseudosteps)

    def _rhs_with_dts(self, t, *args, c, level=0, passmg=False):
        idxcurr = self._dualidxcurr
        mgregidx, srcidx = self._mg_regidx, self._source_regidx
        kerns, banks = self._get_axnpby_kerns, self._prepare_reg_banks

        # Compute -∇·f
        self.dualsystems[level].rhs(t, *args)

        # Coefficients for the dual-time source term
        svals = [c*sc for sc in self._dual_time_source]

        # Source addition -∇·f - dQ/dt
        axnpby = kerns(len(svals) + 1, subdims=self._subdims, level=level)
        banks(args[1], idxcurr[level], *srcidx, level=level)
        self._queue % axnpby(1, *svals)

        if self.multigrid and level != 0 and passmg is not True:
            # Include residual defects at lower polynomial multigrid levels
            axnpby = kerns(2, subdims=self._subdims, level=level)
            banks(args[1], mgregidx[0], level=level)
            self._queue % axnpby(1, -1)

    def finalise_step(self, currsoln, level=0):
        pnreg, regidx = self._pseudo_stepper_nregs, self._dualregidx
        dsrc, srcidx = self._dual_time_source, self._source_regidx

        # Rotate the source registers to the right by one
        regidx[pnreg:pnreg + len(dsrc) - 1] = (srcidx[-1:] + srcidx[:-1])

        # Copy the current soln into the first source register
        self._add(0, regidx[pnreg], 1, currsoln, level=level)


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

    def step(self, t, dt, dtau, level=0):
        add = self._add
        rhs = self._rhs_with_dts
        r0, r1 = self._stepper_regidx

        if r0 != self._dualidxcurr[level]:
            r0, r1 = r1, r0

        rhs(t, r0, r1, c=1/dt, level=level)
        add(0, r1, 1, r0, dtau, r1, level=level)

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

    def step(self, t, dt, dtau, level=0):
        add = self._add
        rhs = self._rhs_with_dts

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time
        r0, r1, r2 = self._stepper_regidx

        # Ensure r0 references the bank containing u(n+1,m)
        if r0 != self._dualidxcurr[level]:
            r0, r1 = r1, r0

        # First stage;
        # r2 = -∇·f(r0) - dQ/dt; r1 = r0 + dtau*r2;
        rhs(t, r0, r2, c=1/dt, level=level)
        add(0, r1, 1, r0, dtau, r2, level=level)

        # Second stage;
        # r2 = -∇·f(r1) - dQ/dt; r1 = 3/4*r0 + 1/4*r1 + 1/4*dtau*r2
        rhs(t, r1, r2, c=1/dt, level=level)
        add(1/4, r1, 3/4, r0, dtau/4, r2, level=level)

        # Third stage;
        # r2 = -∇·f(r1) - dQ/dt; r1 = 1/3*r0 + 2/3*r1 + 2/3*dtau*r2 -
        rhs(t, r1, r2, c=1/dt, level=level)
        add(2/3, r1, 1/3, r0, 2*dtau/3, r2, level=level)

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

    def step(self, t, dt, dtau, level=0):
        add = self._add
        rhs = self._rhs_with_dts

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time
        r0, r1, r2 = self._stepper_regidx

        # Ensure r0 references the bank containing u(n+1,m)
        if r0 != self._dualidxcurr[level]:
            r0, r1 = r1, r0

        # First stage; r1 = -∇·f(r0) - dQ/dt;
        rhs(t, r0, r1, c=1/dt, level=level)

        # Second stage; r2 = r0 + dtau/2*r1; r2 = -∇·f(r2) - dQ/dt;
        add(0, r2, 1, r0, dtau/2, r1, level=level)
        rhs(t, r2, r2, c=1/dt, level=level)

        # As no subsequent stages depend on the first stage we can
        # reuse its register to start accumulating the solution with
        # r1 = r0 + dtau/6*r1 + dtau/3*r2
        add(dtau/6, r1, 1, r0, dtau/3, r2, level=level)

        # Third stage; here we reuse the r2 register
        # r2 = r0 + dtau/2*r2 - dtau/2*dQ/dt
        # r2 = -∇·f(r2) - dQ/dt;
        add(dtau/2, r2, 1, r0, level=level)
        rhs(t, r2, r2, c=1/dt, level=level)

        # Accumulate; r1 = r1 + dtau/3*r2
        add(1, r1, dtau/3, r2, level=level)

        # Fourth stage; again we reuse r2
        # r2 = r0 + dtau*r2
        # r2 = -∇·f(r2) - dQ/dt;
        add(dtau, r2, 1, r0, level=level)
        rhs(t, r2, r2, c=1/dt, level=level)

        # Final accumulation r1 = r1 + dtau/6*r2 = u(n+1,m+1)
        add(1, r1, dtau/6, r2, level=level)

        # Return the index of the bank containing u(n+1,m+1)
        return r1, r0
