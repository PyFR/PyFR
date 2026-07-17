import math

import numpy as np

from pyfr.integrators.implicit.base import BaseImplicitIntegrator
from pyfr.integrators.implicit.nonlinear import StageStats
from pyfr.integrators.registers import (DynamicScalarRegister,
                                        DynamicVectorRegister, VectorRegister)


class BaseImplicitStepper(BaseImplicitIntegrator):
    pass


class BaseSDIRKStepper(BaseImplicitStepper):
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

        # Per-step cache of norms used to damp the predictors
        self._guess_norms = {}

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
            # Explicit stage; no initial guess is required
            if Ai[i] == 0:
                weights.append(None)
            # No prior stages, linearly extrapolate from previous time step
            elif i == 0:
                weights.append((self.c[i], [None]))
            # One prior stage; first-order through its derivative
            elif i == 1:
                weights.append((self.c[i], [[1]]))
            # Prior stages; smoothly extrapolate, then first-order fallback
            else:
                w = [pfit(self.c[:i], np.arange(i) == j, i - 1)(self.c[i])
                     for j in range(i)]
                weights.append((self.c[i], [w, [0]*(i - 1) + [1]]))

        return weights

    def _compute_stage_residual(self, u_n, r_f_prev, u_i, f_i, dt, result):
        pairs = [(1, u_i), (-1, u_n), (-dt*self._gamma, f_i)]
        pairs += [(-dt*Aij, fj)
                  for Aij, fj in zip(self.A[len(r_f_prev)], r_f_prev)]

        self._addv_nz(result, pairs)

    def _compute_stage_initial_guess(self, stage, t_i, u_n, f_prev_list,
                                     f_reg, dt, u_i_reg, residual_fn):
        c_i, wcands = self._guess_weights[stage]
        norms = self._guess_norms

        # Reference norms for damping; these are constant over a step
        f_ref = self._r_f[-1] if wcands[0] is None else f_prev_list[0]
        for r in (f_ref, u_n):
            if r not in norms:
                norms[r] = self._norm2(r)

        # Damp dt to bound the predictor for large time steps
        inc = c_i*dt*norms[f_ref]
        u_norm = norms[u_n]
        adt = dt*u_norm / (u_norm + inc) if u_norm + inc else dt

        # Try each predictor in turn, ending with the trivial guess
        for w in [*wcands, []]:
            # If we don't have weights then use a forward Euler predictor
            if w is None:
                self._add(0, u_i_reg, 1, u_n, c_i*adt, self._r_f[-1])
            # Lagrange interpolation: u = u_n + c_i * dt * sum_j(w_j * f_j)
            else:
                pairs = [(1, u_n)]
                pairs += [(wj*c_i*adt, fj) for wj, fj in zip(w, f_prev_list)]
                self._addv_nz(u_i_reg, pairs)

            rnorm = self._residual_norm(t_i, u_i_reg, f_reg, residual_fn)
            if math.isfinite(rnorm):
                break

        return rnorm

    def step(self, t, dt):
        r_f = self._r_f
        r_un, r_ui = self._r_u

        # Invalidate the predictor damping norm cache
        self._guess_norms.clear()

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

                def initial_guess_fn(u, t_i=t_i, f_reg=f_reg, stage=i,
                                     un=r_un, fprev=f_prev):
                    return self._compute_stage_initial_guess(
                        stage, t_i, un, fprev, f_reg, dt, u, residual_fn
                    )

                stats = self._stage_solve(
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
        pairs = [(dt*ei, fi) for ei, fi in zip(self._err_coeffs, r_f)]
        self._addv_nz(self._r_err, pairs)


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

    _gamma = 1 - 2**-0.5
    _w = 2**0.5 / 4

    A = [[0, 0, 0],
         [_gamma, _gamma, 0],
         [_w, _w, _gamma]]
    b = [_w, _w, _gamma]


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
