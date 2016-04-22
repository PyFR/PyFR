# -*- coding: utf-8 -*-

from pyfr.integrators.dual.base import BaseDualIntegrator


class BaseDualPseudoStepper(BaseDualIntegrator):
    def collect_stats(self, stats):
        super().collect_stats(stats)

        stats.set('solver-time-integrator', 'nsteps', self.nsteps)
        stats.set('solver-time-integrator', 'nfevals', self._stepper_nfevals)

    def _add_dual_source(self, dt, rhs, currsoln):
        # Coefficients for the dual-time source term
        coeffs = [c/dt for c in self._dual_time_source]

        # Get a suitable set of axnpby kernels
        axnpby = self._get_axnpby_kerns(len(coeffs) + 1,
                                        subdims=self._subdims)

        # Prepare the matrix banks
        self._prepare_reg_banks(rhs, currsoln, *self._source_regidx)

        # Bind and run the axnpby kernels
        self._queue % axnpby(1.0, *coeffs)

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
        add, add_dual_source = self._add, self._add_dual_source
        rhs = self.system.rhs
        ut, f = self._stepper_regidx

        rhs(t, ut, f)
        add_dual_source(dt, f, ut)
        add(1.0, ut, dtau, f)

        return ut


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
        add, add_dual_source = self._add, self._add_dual_source
        rhs = self.system.rhs

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time
        r0, r1, r2 = self._stepper_regidx

        # Ensure r0 references the bank containing u(n+1,m)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage;
        # r2 = -∇·f(r0) - dQ/dt; r1 = r0 + dtau*r2
        rhs(t, r0, r2)
        add_dual_source(dt, r2, r0)
        add(0.0, r1, 1.0, r0, dtau, r2)

        # Second stage;
        # r2 = -∇·f(r1) - dQ/dt; r1 = 0.75*r0 + 0.25*r1 + 0.25*dtau*r2
        rhs(t, r1, r2)
        add_dual_source(dt, r2, r0)
        add(0.25, r1, 0.75, r0, dtau/4.0, r2)

        # Third stage;
        # r2 = -∇·f(r1) - dQ/dt; r1 = 1.0/3.0*r0 + 2.0/3.0*r1 + 2.0/3.0*dtau*r2
        rhs(t, r1, r2)
        add_dual_source(dt, r2, r1)
        add(2.0/3.0, r1, 1.0/3.0, r0, 2.0*dtau/3.0, r2)

        # Return the index of the bank containing u(n+1,m+1)
        return r1


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
        add, add_dual_source  = self._add, self._add_dual_source
        rhs, pi_coeff = self.system.rhs, self._point_implicit_coeff

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time
        r0, r1, r2 = self._stepper_regidx

        # Ensure r0 references the bank containing u(n+1,m)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage; r1 = -∇·f(r0) - dQ/dt
        rhs(t, r0, r1)
        add_dual_source(dt, r1, r0)

        # Second stage; r2 = r0 + dtau/2*r1; r2 = -∇·f(r2) - dQ/dt
        add(0.0, r2, 1.0, r0, dtau/2.0, r1)
        rhs(t, r2, r2)
        add_dual_source(dt, r2, r0)

        # As no subsequent stages depend on the first stage we can
        # reuse its register to start accumulating the solution with
        # r1 = r0 + pi_coeff*r1 + pi_coeff*r2
        add(pi_coeff(dt, dtau/6.0), r1, 1.0, r0, pi_coeff(dt, dtau/3.0), r2)

        # Third stage; here we reuse the r2 register
        # r2 = r0 + dtau/2*r2
        # r2 = -∇·f(r2)
        add(dtau/2.0, r2, 1.0, r0)
        rhs(t, r2, r2)
        add_dual_source(dt, r2, r0)

        # Accumulate; r1 = r1 + pi_coeff*r2
        add(1.0, r1, pi_coeff(dt, dtau/3.0), r2)

        # Fourth stage; again we reuse r2
        # r2 = r0 + dtau*r2
        # r2 = -∇·f(r2) - dQ/dt
        add(dtau, r2, 1.0, r0)
        rhs(t, r2, r2)
        add_dual_source(dt, r2, r0)

        # Final accumulation r1 = r1 + pi_coeff*r2 = u(n+1,m+1)
        add(1.0, r1, pi_coeff(dt, dtau/6.0), r2)

        # Return the index of the bank containing u(n+1,m+1)
        return r1
