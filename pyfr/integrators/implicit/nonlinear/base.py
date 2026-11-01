from collections import namedtuple
import math
import time

from pyfr.integrators.implicit.base import BaseImplicitIntegrator
from pyfr.integrators.registers import DynamicScalarRegister, ScalarRegister
from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll


# Statistics from a nonlinear solve
SolveStats = namedtuple('SolveStats',
                        'nonlin_iters precond_apps init_resid final_resid '
                        'converged stalled nmatvec nretries inner_tol probe',
                        defaults=('',)*4)

# Solve statistics for a stage along with its cost and preconditioner state
StageStats = namedtuple('StageStats',
                        ['stage', *SolveStats._fields, 'nrhs', 'wtime',
                         'precond_gdt_ratio', 'precond_built',
                         'precond_wtime', 'precond_nbuilds'])


class NonlinearDivergenceError(Exception):
    pass


class BaseNonlinearSolver(BaseImplicitIntegrator):
    nonlinear_name = None

    # Columns of the per-stage statistics
    stage_stats_fields = StageStats._fields

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

    def _stage_solve(self, stage, t, u_reg, f_reg, residual_fn,
                     initial_guess_fn, gamma_dt):
        comm, _, _ = get_comm_rank_root()
        pc = self._preconditioner

        # Note the RHS evaluation and preconditioner build counters
        nrhs = self.nrhsevals
        pc_wtime, nbuilds = pc.build_wtime_total, pc.nbuilds

        # Run the core solve
        tstart = time.perf_counter() - self.backend.kernel_setup_wtime
        stats = self._solve(t, u_reg, f_reg, residual_fn, initial_guess_fn,
                            gamma_dt)
        wtime = time.perf_counter() - self.backend.kernel_setup_wtime - tstart

        # Agree on the stage time across ranks
        wtime = scal_coll(comm.Allreduce, wtime, op=mpi.MAX)

        # Split the time between the solve and any build
        pc_wtime = pc.build_wtime_total - pc_wtime
        nrhs = self.nrhsevals - nrhs
        wtime = max(wtime - pc_wtime, 0.0)

        # Note if the preconditioner was rebuilt and how gamma*dt has drifted
        precond_built = int(pc.nbuilds > nbuilds)
        gdt_ratio = gamma_dt / pc.gdt_built if pc.gdt_built else 1.0

        return StageStats(stage, *stats, nrhs, wtime, gdt_ratio,
                          precond_built, pc_wtime, pc.nbuilds)
