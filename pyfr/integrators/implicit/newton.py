from collections import namedtuple
import math
import time

from pyfr.integrators.implicit.krylov import BaseKrylovSolver
from pyfr.integrators.registers import ScalarRegister
from pyfr.mpiutil import get_comm_rank_root


StageStats = namedtuple('StageStats',
                        'stage niters nkrylov nprecond resid0 resid ktol')


class NewtonDivergenceError(Exception):
    pass


class NewtonSolver(BaseKrylovSolver):
    _newton_resid = ScalarRegister(rhs=False)
    _jfnk_temp = ScalarRegister()

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        sect = 'solver-time-integrator'
        b = lambda k, d: cfg.getbool(sect, f'newton-{k}', d)
        f = lambda k, d: cfg.getfloat(sect, f'newton-{k}', d)
        i = lambda k, d: cfg.getint(sect, f'newton-{k}', d)
        h = lambda k: cfg.hasopt(sect, f'newton-{k}')

        self._newton_rtol = f('rtol', 1e-4)
        self._newton_atol = f('atol', 1e-8)
        self._newton_maxiter = i('max-iter', 10)

        if self._newton_rtol < 10*backend.fpdtype_eps:
            raise ValueError('Newton relative tolerance too small for '
                             'current precision')

        # Get variable names for per-variable tolerances
        convars = systemcls.elementscls.convars(mesh.ndims, cfg)

        # Check for per-variable atol
        self._use_weighted_atol = any(h(f'atol-{v}') for v in convars)

        # Build per-variable atol list
        self._newton_atols = [f(f'atol-{v}', self._newton_atol)
                              for v in convars]

        # For weighted norm, atol is already baked in
        if self._use_weighted_atol:
            self._newton_atol = 1.0

        # Scaling factors for Newton
        min_atol = min(self._newton_atols)
        self._scales = tuple(a / min_atol for a in self._newton_atols)
        self._inv_scales = tuple(min_atol / a for a in self._newton_atols)

        super().__init__(backend, systemcls, mesh, initsoln, cfg)

    def _calc_rnorm(self, r):
        weights = self._newton_atols if self._use_weighted_atol else ()
        return self._norm2(r, weights=weights, norm_gndofs=True)

    def _jfnk_matvec(self, t, u, f, gamma_dt, eps, v_s, result):
        self._add(0, self._jfnk_temp, 1, u, eps, v_s,
                  in_scale=self._scales, in_scale_idxs=(2,))
        self._rhs(t, self._jfnk_temp, self._jfnk_temp)
        self._add(0, result, 1, v_s, -gamma_dt/eps, self._jfnk_temp,
                  gamma_dt/eps, f, in_scale=self._scales, in_scale_idxs=(1,),
                  out_scale=self._inv_scales)

    def _newton_iterate(self, t, u_reg, f_reg, gamma_dt, residual_fn,
                        initial_guess_fn):
        # Helper function to compute the residual norm
        def calc_rnorm():
            self._rhs(t, u_reg, f_reg)
            residual_fn(u_reg, f_reg, self._newton_resid)
            return self._calc_rnorm(self._newton_resid)

        # Helper function to compute the matrix-vector product
        def matvec(v, result):
            self._jfnk_matvec(t, u_reg, f_reg, gamma_dt, self._krylov_eps, v,
                              result)

        initial_guess_fn(u_reg)
        krylov_total = precond_total = 0
        rnorm = None

        for i in range(self._newton_maxiter):
            # Ensure we have a valid (scaled) residual norm
            if rnorm is None:
                rnorm = calc_rnorm()

            if not math.isfinite(rnorm):
                raise NewtonDivergenceError('Non-finite residual')

            # Set the relative tolerance based on the initial residual
            if i == 0:
                rnorm_init = rnorm
                tol = max(rnorm*self._newton_rtol, self._newton_atol)

            # Check for convergence
            if rnorm < tol:
                break

            # Scale the residual vector for the Krylov solver
            self._add(1, self._newton_resid, out_scale=self._inv_scales)

            niters, nprecond = self._krylov_solve(
                matvec, self._newton_resid, u_reg, None,
                accumulate=True, accumulate_scale=self._scales
            )
            rnorm = None

            krylov_total += niters
            precond_total += nprecond
        # If we failed to converge ensure we have a valid residual
        else:
            if rnorm is None:
                rnorm = calc_rnorm()

        return i + 1, krylov_total, precond_total, rnorm_init, rnorm, tol

    def _newton_stage_solve(self, t, u_reg, f_reg, residual_fn,
                            initial_guess_fn, gamma_dt):
        comm, rank, root = get_comm_rank_root()

        for i in range(self._tol_controller.max_retries + 1):
            # Determine and broadcast the optimal Krylov tolerance
            if rank == root:
                krylov_tol = self._tol_controller.select_tolerance()
                t_start = time.perf_counter()
            else:
                krylov_tol = None

            self._krylov_rtol = comm.bcast(krylov_tol, root=root)

            *stats, rnorm, tol = self._newton_iterate(
                t, u_reg, f_reg, gamma_dt, residual_fn, initial_guess_fn
            )

            # Have the root rank update the tolerance controller
            if rank == root:
                wall_time = time.perf_counter() - t_start
                self._tol_controller.update(wall_time, gamma_dt,
                                            rnorm < tol)

            if rnorm < tol or i == self._tol_controller.max_retries:
                break

        return (*stats, rnorm, krylov_tol)
