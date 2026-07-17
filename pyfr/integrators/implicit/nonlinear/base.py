from collections import namedtuple
import math

from pyfr.integrators.implicit.base import BaseImplicitIntegrator
from pyfr.integrators.registers import DynamicScalarRegister, ScalarRegister


StageStats = namedtuple('StageStats',
                        'stage niters nmatvec nprecond resid0 resid inner_tol '
                        'precond_gdt_ratio precond_built')


class NonlinearDivergenceError(Exception):
    pass


class BaseNonlinearSolver(BaseImplicitIntegrator):
    nonlinear_name = None

    _nl_resid = ScalarRegister(rhs=False)
    _nl_temp = ScalarRegister()
    _nl_delta = DynamicScalarRegister(rhs=False, extent='solve')

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        sect = 'solver-time-integrator'
        b = lambda k, d: cfg.getbool(sect, f'nonlinear-{k}', d)
        f = lambda k, d: cfg.getfloat(sect, f'nonlinear-{k}', d)
        i = lambda k, d: cfg.getint(sect, f'nonlinear-{k}', d)
        h = lambda k: cfg.hasopt(sect, f'nonlinear-{k}')

        self._nl_rtol = f('rtol', 1e-4)
        self._nl_atol = f('atol', 1e-8)
        self._nl_maxiter = i('max-iter', 10)

        if self._nl_rtol < 10*backend.fpdtype_eps:
            raise ValueError('Nonlinear relative tolerance too small for '
                             'current precision')

        # Get variable names for per-variable tolerances
        convars = systemcls.elementscls.convars(mesh.ndims, cfg)

        # Check for per-variable atol
        self._use_weighted_atol = any(h(f'atol-{v}') for v in convars)

        # Build per-variable atol list
        self._nl_atols = [f(f'atol-{v}', self._nl_atol)
                              for v in convars]

        # For weighted norm, atol is already baked in
        if self._use_weighted_atol:
            self._nl_atol = 1.0

        # Scaling factors for the nonlinear solve
        min_atol = min(self._nl_atols)
        self._scales = tuple(a / min_atol for a in self._nl_atols)
        self._inv_scales = tuple(min_atol / a for a in self._nl_atols)

        # Configure the shared line-search globalisation
        self._linesearch = b('linesearch', False)
        if self._linesearch:
            self._linesearch_maxiter = i('linesearch-max-iter', 5)
            self._linesearch_fact = f('linesearch-fact', 0.5)
            self._linesearch_c1 = f('linesearch-c1', 1e-4)
            self._size_register(self._nl_delta, 1)

        super().__init__(backend, systemcls, mesh, initsoln, cfg)

    def _calc_rnorm(self, r):
        weights = self._nl_atols if self._use_weighted_atol else ()
        return self._norm2(r, weights=weights, norm_gndofs=True)

    def _residual_norm(self, t, u_reg, f_reg, residual_fn):
        self._rhs(t, u_reg, f_reg)
        residual_fn(u_reg, f_reg, self._nl_resid)
        return self._calc_rnorm(self._nl_resid)

    def _line_search(self, t, u_reg, f_reg, delta_reg, residual_fn, rnorm_old):
        alpha = alpha_best = 1.0
        rnorm_best = math.inf

        for _ in range(self._linesearch_maxiter):
            self._add(0, self._nl_temp, 1, u_reg, alpha, delta_reg,
                      in_scale=self._scales, in_scale_idxs=(2,))

            self._rhs(t, self._nl_temp, f_reg)
            residual_fn(self._nl_temp, f_reg, self._nl_resid)
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
                raise NonlinearDivergenceError('Non-finite residual in line '
                                               'search')

            alpha = alpha_best

        # Apply the update
        self._add(1, u_reg, alpha, delta_reg, in_scale=self._scales,
                  in_scale_idxs=(1,))

        # Recompute the residual
        return self._residual_norm(t, u_reg, f_reg, residual_fn)

    def _stage_solve(self, t, u_reg, f_reg, residual_fn, initial_guess_fn,
                     gamma_dt):
        # Run the core solve, detecting precond rebuild and gamma*dt drift
        pc_built_before = self._precond_computed

        stats = self._solve(t, u_reg, f_reg, residual_fn, initial_guess_fn,
                            gamma_dt)

        precond_built = not pc_built_before and self._precond_computed
        gdt_built = self._precond_gdt_built
        gdt_ratio = gamma_dt / gdt_built if gdt_built > 0 else 1.0

        return (*stats, gdt_ratio, precond_built)
