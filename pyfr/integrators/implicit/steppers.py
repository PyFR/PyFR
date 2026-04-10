import numpy as np

from pyfr.integrators.implicit.base import BaseImplicitIntegrator
from pyfr.integrators.implicit.newton import NewtonSolver, StageStats
from pyfr.integrators.registers import (DynamicScalarRegister,
                                        DynamicVectorRegister, VectorRegister)


class BaseImplicitStepper(BaseImplicitIntegrator):
    pass


class BaseSDIRKStepper(BaseImplicitStepper, NewtonSolver):
    A = []
    b = []
    bhat = []
    _gamma = 0

    stepper_order = 1

    _r_u = VectorRegister(n=2)
    _r_f = DynamicVectorRegister()
    _r_err = DynamicScalarRegister(rhs=False)

    def __init__(self, *args, **kwargs):
        self._nstages = len(self.b)
        self._size_register(self._r_f, self._nstages)

        self.c = [sum(row) for row in self.A]

        if self.bhat:
            self._size_register(self._r_err, 1)
            self._err_coeffs = [b - bh for b, bh in zip(self.b, self.bhat)]

        # Precompute interpolation weights for initial guesses
        self._guess_weights = self._compute_guess_weights()

        self._fsal = self.A[0][0] == 0 and self.A[-1] == self.b
        self._fsal_valid = False

        super().__init__(*args, **kwargs)

    @property
    def stepper_has_errest(self):
        return self.controller_needs_errest and len(self.bhat)

    def _compute_guess_weights(self):
        pfit = np.polynomial.Polynomial.fit
        weights = []

        for i, Ai in enumerate(self.A):
            # Explicit stage; no interpolation needed
            if Ai[i] == 0:
                weights.append(None)
            # No prior stages, linearly extrapolate from previous time step
            elif i == 0:
                weights.append((self.c[i], None))
            # Prior stages; smoothly extrapolate from prior stages
            else:
                w = [pfit(self.c[:i], np.arange(i) == j, i - 1)(self.c[i])
                     for j in range(i)]
                weights.append((self.c[i], w))

        return weights

    def _compute_stage_residual(self, u_n, r_f_prev, u_i, f_i, dt, result):
        coeffs = [0, result, 1, u_i, -1, u_n, -dt*self._gamma, f_i]
        for Aij, fj in zip(self.A[len(r_f_prev)], r_f_prev):
            if Aij != 0:
                coeffs.extend([-dt*Aij, fj])

        self._addv(coeffs[::2], coeffs[1::2])

    def _compute_stage_initial_guess(self, stage, u_n, f_prev_list, dt,
                                     u_i_reg):
        c_i, w = self._guess_weights[stage]

        # Damp dt to bound the predictor for large time steps
        f_ref = self._r_f[-1] if w is None else f_prev_list[0]
        inc = c_i*dt*self._norm2(f_ref)
        u_norm = self._norm2(u_n)
        adt = dt*u_norm / (u_norm + inc) if u_norm + inc else dt

        # If we don't have weights then use a forward Euler predictor
        if w is None:
            self._add(0, u_i_reg, 1, u_n, c_i*adt, self._r_f[-1])
        # Lagrange interpolation: u = u_n + c_i * dt * sum_j(w_j * f_j)
        else:
            coeffs = [0, u_i_reg, 1, u_n]
            for wj, fj in zip(w, f_prev_list):
                if wj:
                    coeffs.extend([wj*c_i*adt, fj])
            self._addv(coeffs[::2], coeffs[1::2])

    def step(self, t, dt):
        r_f = self._r_f
        r_un, r_ui = self._r_u

        # Ensure r_un references the bank containing u(t)
        if r_un != self.idxcurr:
            r_un, r_ui = r_ui, r_un

        # Precompute f(t, u_n) for the stage-0 initial guess when implicit
        if self.A[0][0] != 0:
            self._rhs(t, r_un, r_f[-1])

        for i, (Ai, ci) in enumerate(zip(self.A, self.c)):
            t_i = t + ci*dt
            f_reg = r_f[i]

            if Ai[i] == 0:
                if i != 0 or not self._fsal_valid:
                    self._rhs(t_i, r_un, f_reg)
            else:
                f_prev = r_f[:i]

                def residual_fn(u, f, result, un=r_un, fprev=f_prev):
                    self._compute_stage_residual(un, fprev, u, f, dt, result)

                def initial_guess_fn(u, stage=i, un=r_un, fprev=f_prev):
                    self._compute_stage_initial_guess(stage, un, fprev, dt, u)

                stats = self._newton_stage_solve(
                    t_i, r_ui, f_reg, residual_fn, initial_guess_fn,
                    self._gamma*dt
                )
                self._stage_stats.append(StageStats(i, *stats))

                if i < self._nstages - 1:
                    self._rhs(t_i, r_ui, f_reg)

        # Handle FSAL
        if self._fsal:
            r_f[0], r_f[-1] = r_f[-1], r_f[0]
            self._fsal_valid = True

        # Compute error estimate if this scheme has an embedded pair
        if self.stepper_has_errest:
            self._compute_error_estimate(dt, r_f)
            return r_ui, r_un, self._r_err

        return r_ui

    def _compute_error_estimate(self, dt, r_f):
        coeffs = [0, self._r_err]
        for ei, fi in zip(self._err_coeffs, r_f):
            if ei != 0:
                coeffs.extend([dt * ei, fi])

        self._addv(coeffs[::2], coeffs[1::2])


class ImplicitEulerStepper(BaseSDIRKStepper):
    stepper_name = 'euler'
    stepper_order = 1
    _gamma = 1.0
    A = [[_gamma]]
    b = [1.0]


class TrapeziumStepper(BaseSDIRKStepper):
    stepper_name = 'trapezium'
    stepper_order = 2
    _gamma = 0.5
    A = [[0.0, 0.0],
         [_gamma, _gamma]]
    b = [_gamma, _gamma]



class TRBDF2Stepper(BaseSDIRKStepper):
    stepper_name = 'trbdf2'
    stepper_order = 2

    _gamma = 2 - 2**0.5
    _d = _gamma / 2
    _w = 2**0.5 / 4

    A = [[0, 0, 0],
         [_d, _d, 0],
         [_w, _w, _d]]
    b = [_w, _w, _d]


class Kvaerno43Stepper(BaseSDIRKStepper):
    stepper_name = 'kvaerno43'
    stepper_order = 3

    _gamma = 0.4358665215

    A = [[0, 0, 0, 0],
         [_gamma, _gamma, 0, 0],
         [0.490563388419108, 0.073570090080892, _gamma, 0],
         [0.308809969973036, 1.490563388254106, -1.235239879727145, _gamma]]
    b = A[-1]

    # 2nd order embedded method for error estimation
    bhat = [0.490563388419108, 0.073570090080892, _gamma, 0]


class ESDIRK32aStepper(BaseSDIRKStepper):
    stepper_name = 'esdirk32a'
    stepper_order = 3

    _gamma = 0.4358665215084590

    A = [[0, 0, 0, 0],
         [_gamma, _gamma, 0, 0],
         [0.1406888706327204, 0.5986277712457818, _gamma, 0],
         [0.1023994984498115, 0.3694244393719295, 0.0923096091397000, _gamma]]

    b = A[-1]
    bhat = [0.1571093680257579, 0.3413898519498105, 0.0657341406893316,
            0.4357666393351000]
