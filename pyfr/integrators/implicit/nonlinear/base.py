import math

from pyfr.integrators.implicit.base import (BaseImplicitIntegrator,
                                            DivergenceError)
from pyfr.integrators.registers import DynamicScalarRegister, ScalarRegister


class BaseNonlinearSolver(BaseImplicitIntegrator):
    nonlinear_name = None

    _nl_resid = ScalarRegister(rhs=False)
    _nl_delta = DynamicScalarRegister(rhs=False, extent='solve')

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        sect = 'solver-time-integrator'
        b = lambda k, d: cfg.getbool(sect, f'nonlinear-{k}', d)
        f = lambda k, d: cfg.getfloat(sect, f'nonlinear-{k}', d)
        i = lambda k, d: cfg.getint(sect, f'nonlinear-{k}', d)

        self._nl_rtol = f('rtol', 1e-4)
        self._nl_maxiter = i('max-iter', 10)

        # Iterations and per-iteration reduction which define a stall
        self._nl_stall_iters = i('stall-iters', 2)
        self._nl_stall_ratio = f('stall-ratio', 0.5)

        # Treat a stalled residual within this factor of the tolerance as met
        self._nl_stall_fact = f('stall-fact', 2.0)

        # Give up on a solve once the residual has grown by this factor
        self._nl_diverge_fact = f('diverge-fact', 10.0)

        # Tighten the Krylov tolerance by this factor on each retry
        self._retry_tol_fact = cfg.getfloat(sect, 'linear-retry-fact', 0.1)

        if self._nl_rtol < 10*backend.fpdtype_eps:
            raise ValueError('Nonlinear relative tolerance too small for '
                             'current precision')

        # Configure the shared line-search globalisation
        self._linesearch = b('linesearch', False)
        if self._linesearch:
            self._linesearch_maxiter = i('linesearch-max-iter', 5)
            self._linesearch_fact = f('linesearch-fact', 0.5)
            self._linesearch_c1 = f('linesearch-c1', 1e-4)
            self._size_register(self._nl_delta, 1)

        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        # For weighted norms the atol is already baked into the scales
        self._nl_atol = 1.0 if self._use_weighted_atol else self._solve_atol

    def _nl_tol(self, rnorm0):
        return max(rnorm0*self._nl_rtol, self._nl_atol)

    def _nl_stalled(self, rhist, rnorm):
        # A stall is little progress over the last few iterations
        if len(rhist) < self._nl_stall_iters:
            return False
        else:
            ref = rhist[-self._nl_stall_iters]
            return rnorm > self._nl_stall_ratio**self._nl_stall_iters*ref

    def _nl_diverged(self, rnorm_init, rnorm):
        return rnorm > self._nl_diverge_fact*rnorm_init

    def _nl_outcome(self, rnorm_init, rnorm, tol):
        # Converged, or else accepted as stalled unless it has diverged
        converged = rnorm < tol
        stalled = not converged and not self._nl_diverged(rnorm_init, rnorm)

        return converged, stalled

    def _residual_norm(self, t, u_reg, f_reg, residual_fn):
        self._rhs(t, u_reg, f_reg)
        residual_fn(u_reg, f_reg, self._nl_resid)
        return self._calc_rnorm(self._nl_resid)

    def _line_search(self, t, u_reg, f_reg, delta_reg, residual_fn, rnorm_old):
        alpha = alpha_best = 1.0
        rnorm_best = math.inf

        for _ in range(self._linesearch_maxiter):
            self._add(0, self._solve_temp, 1, u_reg, alpha, delta_reg,
                      in_scale=self._scales, in_scale_idxs=(2,))

            self._rhs(t, self._solve_temp, f_reg)
            residual_fn(self._solve_temp, f_reg, self._nl_resid)
            rnorm_new = self._calc_rnorm(self._nl_resid)

            # Note the best trial step; false for non-finite norms
            if rnorm_new < rnorm_best:
                alpha_best, rnorm_best = alpha, rnorm_new

            if rnorm_new <= (1 - self._linesearch_c1*alpha)*rnorm_old:
                break

            alpha *= self._linesearch_fact
        # Failed to satisfy the Armijo condition; take the best finite step
        else:
            if math.isinf(rnorm_best):
                raise DivergenceError('Non-finite residual in line search')

            alpha = alpha_best

        # Apply the update
        self._add(1, u_reg, alpha, delta_reg, in_scale=self._scales,
                  in_scale_idxs=(1,))

        # Recompute the residual
        return self._residual_norm(t, u_reg, f_reg, residual_fn)

    def _stage_solve(self, stage, t, u_reg, f_reg, residual_fn,
                     initial_guess_fn, gamma_dt):
        # Run the core solve
        def solve_fn():
            return self._solve(t, u_reg, f_reg, residual_fn,
                               initial_guess_fn, gamma_dt)

        return self._timed_stage(stage, gamma_dt, solve_fn)
