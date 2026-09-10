import math
import time

from pyfr.integrators.implicit.nonlinear.base import (BaseNonlinearSolver,
                                                      NonlinearDivergenceError)
from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll


class NewtonSolver(BaseNonlinearSolver):
    nonlinear_name = 'newton'
    requires_linear_solver = True

    def _jfnk_matvec(self, t, u, f, gamma_dt, eps, v_s, result):
        self._add(0, self._nl_temp, 1, u, eps, v_s, in_scale=self._scales,
                  in_scale_idxs=(2,))
        self._rhs(t, self._nl_temp, self._nl_temp)
        self._add(0, result, 1, v_s, -gamma_dt/eps, self._nl_temp,
                  gamma_dt/eps, f, in_scale=self._scales, in_scale_idxs=(1,),
                  out_scale=self._inv_scales)

    def _newton_iterate(self, t, u_reg, f_reg, gamma_dt, residual_fn,
                        initial_guess_fn, precond, rtol):
        # Matrix-vector product; eps = eps_base/‖v‖ (pass ‖v‖ to skip norm)
        def matvec(v, result, vnorm=None):
            if vnorm is None:
                vnorm = self._norm2(v)

            self._jfnk_matvec(t, u_reg, f_reg, gamma_dt, self._fd_eps/vnorm,
                              v, result)

        # Pick an initial starting guess
        rnorm = initial_guess_fn(u_reg)

        krylov_total = precond_total = 0

        for i in range(self._nl_maxiter):
            # Ensure we have a valid (scaled) residual norm
            if rnorm is None:
                rnorm = self._residual_norm(t, u_reg, f_reg, residual_fn)

            if not math.isfinite(rnorm):
                raise NonlinearDivergenceError('Non-finite residual')

            # Set the relative tolerance based on the initial residual
            if i == 0:
                rnorm_init = rnorm
                tol = max(rnorm*self._nl_rtol, self._nl_atol)

                # An already converged guess needs no solve (avoids 0 residual)
                if rnorm < tol:
                    break
            # After we've done at least one step check for converge
            elif rnorm < tol:
                break

            # Choose a suitable finite difference perturbation
            self._compute_fd_eps(u_reg)

            # Compute the preconditioner
            self._compute_precond(t, u_reg, gamma_dt, self._rhs, f_reg,
                                  self._nl_temp, eps_scales=self._scales)

            # Scale the residual vector for the Krylov solver
            self._add(1, self._nl_resid, out_scale=self._inv_scales)

            if self._linesearch:
                niters, nprecond = self._krylov_solve(
                    matvec, self._nl_resid, self._nl_delta,
                    precond, rtol=rtol, accumulate=False
                )

                # Line search applies the damped step and returns the new norm
                rnorm = self._line_search(t, u_reg, f_reg, self._nl_delta,
                                          residual_fn, rnorm)
            else:
                niters, nprecond = self._krylov_solve(
                    matvec, self._nl_resid, u_reg, precond, rtol=rtol,
                    accumulate=True, accumulate_scale=self._scales
                )
                rnorm = None

            krylov_total += niters
            precond_total += nprecond
        # If we failed to converge ensure we have a valid residual
        else:
            if rnorm is None:
                rnorm = self._residual_norm(t, u_reg, f_reg, residual_fn)

        return i + 1, krylov_total, precond_total, rnorm_init, rnorm, tol

    def _solve(self, t, u_reg, f_reg, residual_fn, initial_guess_fn, gamma_dt):
        comm, _, _ = get_comm_rank_root()

        # Scaled preconditioner: M̃⁻¹ = S⁻¹ M⁻¹ S
        if self._preconditioner.active:
            def precond(in_reg, out_reg):
                self._apply_precond(in_reg, out_reg, in_scale=self._scales,
                                    out_scale=self._inv_scales)
        else:
            precond = None

        for i in range(self._tol_controller.max_retries + 1):
            pc_built_before_retry = self._precond_computed

            # Select a Krylov tolerance
            krylov_rtol = self._tol_controller.select_tolerance()

            # Iterate
            tstart = time.perf_counter()
            *stats, rnorm, tol = self._newton_iterate(
                t, u_reg, f_reg, gamma_dt, residual_fn, initial_guess_fn,
                precond, krylov_rtol
            )
            dt = time.perf_counter() - tstart
            wtime = scal_coll(comm.Allreduce, dt, op=mpi.MAX)

            built_this_retry = (not pc_built_before_retry and
                                self._precond_computed)
            if not built_this_retry:
                self._tol_controller.update(wtime, gamma_dt, rnorm < tol)

            if rnorm < tol or i == self._tol_controller.max_retries:
                break

        return (*stats, rnorm, krylov_rtol)
