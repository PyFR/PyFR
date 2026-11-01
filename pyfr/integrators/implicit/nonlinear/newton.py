import math
import time

from pyfr.integrators.implicit.nonlinear.base import (BaseNonlinearSolver,
                                                      NonlinearDivergenceError,
                                                      SolveStats)
from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll


class NewtonSolver(BaseNonlinearSolver):
    nonlinear_name = 'newton'
    requires_linear_solver = True

    # Loosest forcing term the oversolving safeguard may select
    _eta_max = 0.9

    def _jfnk_matvec(self, t, u, f, gamma_dt, eps, v_s, result):
        self._add(0, self._nl_temp, 1, u, eps, v_s, in_scale=self._scales,
                  in_scale_idxs=(2,))
        self._rhs(t, self._nl_temp, self._nl_temp)
        self._add(0, result, 1, v_s, -gamma_dt/eps, self._nl_temp,
                  gamma_dt/eps, f, in_scale=self._scales, in_scale_idxs=(1,),
                  out_scale=self._inv_scales)

    def _newton_iterate(self, t, u_reg, f_reg, gamma_dt, residual_fn,
                        initial_guess_fn, rtol, safeguard, rnorm_init=None):
        # Matrix-vector product; eps = eps_base/‖v‖ (pass ‖v‖ to skip norm)
        def matvec(v, result, vnorm=None):
            if vnorm is None:
                vnorm = self._norm2(v)

            self._jfnk_matvec(t, u_reg, f_reg, gamma_dt, self._fd_eps/vnorm,
                              v, result)

        if self._preconditioner.active:
            def precond(in_reg, out_reg):
                self._apply_precond(in_reg, out_reg, in_scale=self._scales,
                                    out_scale=self._inv_scales)
        else:
            precond = None

        # Pick an initial starting guess
        rnorm = initial_guess_fn(u_reg)

        niters = krylov_total = precond_total = 0
        rhist = []

        for i in range(self._nl_maxiter):
            # Ensure we have a valid (scaled) residual norm
            if rnorm is None:
                rnorm = self._residual_norm(t, u_reg, f_reg, residual_fn)

            if not math.isfinite(rnorm):
                raise NonlinearDivergenceError('Non-finite residual')

            # Set the relative tolerance based on the initial residual
            if i == 0:
                if rnorm_init is None:
                    rnorm_init = rnorm

                tol = self._nl_tol(rnorm_init)

                # An already converged guess needs no solve (avoids 0 residual)
                if rnorm < tol:
                    break
            # After we've done at least one step check for converge
            elif rnorm < tol:
                break
            # Give up once the residual has grown well beyond its initial value
            elif self._nl_diverged(rnorm_init, rnorm):
                break
            # Stop at a stalled residual; near the tolerance it has converged
            elif self._nl_stalled(rhist, rnorm):
                tol *= self._nl_stall_fact
                break

            rhist.append(rnorm)

            # Choose a suitable finite difference perturbation
            self._compute_fd_eps(u_reg)

            # Compute the preconditioner
            self._compute_precond(t, u_reg, gamma_dt, self._rhs, f_reg,
                                  self._nl_temp, eps_scales=self._scales)

            # Scale the residual vector for the Krylov solver
            self._add(1, self._nl_resid, out_scale=self._inv_scales)

            # Solve no more tightly than this iteration needs to finish
            if safeguard:
                ktol = min(max(rtol, 0.5*tol / rnorm), self._eta_max)
            else:
                ktol = rtol

            if self._linesearch:
                nkrylov, nprecond = self._krylov_solve(
                    matvec, self._nl_resid, self._nl_delta,
                    precond, rtol=ktol, accumulate=False
                )

                # Line search applies the damped step and returns the new norm
                rnorm = self._line_search(t, u_reg, f_reg, self._nl_delta,
                                          residual_fn, rnorm)
            else:
                nkrylov, nprecond = self._krylov_solve(
                    matvec, self._nl_resid, u_reg, precond, rtol=ktol,
                    accumulate=True, accumulate_scale=self._scales
                )
                rnorm = None

            niters += 1
            krylov_total += nkrylov
            precond_total += nprecond
        # If we failed to converge ensure we have a valid residual
        else:
            if rnorm is None:
                rnorm = self._residual_norm(t, u_reg, f_reg, residual_fn)

        converged, stalled = self._nl_outcome(rnorm_init, rnorm, tol)

        return SolveStats(niters, precond_total, rnorm_init, rnorm,
                          int(converged), int(stalled), krylov_total)

    def _solve(self, t, u_reg, f_reg, residual_fn, initial_guess_fn, gamma_dt):
        comm, _, _ = get_comm_rank_root()

        # Residual norm of the current iterate for warm-started retries
        def warm_guess_fn(u):
            return self._residual_norm(t, u, f_reg, residual_fn)

        pc_built_before = self._precond_computed
        guess_fn, rnorm_init = initial_guess_fn, None
        niters = nmatvec = nprecond = 0

        # Select a Krylov tolerance for this operator and preconditioner
        build = self._preconditioner.nbuilds
        krylov_rtol = self._tol_controller.select_tolerance(gamma_dt, build)
        probing = self._tol_controller.probing

        # Treat the tolerance as a cap unless it is fixed or a retry
        safeguard = self._tol_controller.safeguard

        tstart = time.perf_counter() - self.backend.kernel_setup_wtime

        for i in range(self._tol_controller.max_retries + 1):
            # Iterate
            stats = self._newton_iterate(
                t, u_reg, f_reg, gamma_dt, residual_fn, guess_fn,
                krylov_rtol, safeguard, rnorm_init
            )
            rnorm_init = stats.init_resid

            # Accumulate the work done by this attempt
            niters += stats.nonlin_iters
            nmatvec += stats.nmatvec
            nprecond += stats.precond_apps

            # Stop once converged or out of retries
            if stats.converged or i == self._tol_controller.max_retries:
                break

            # A tighter tolerance cannot help once Krylov is at its budget
            if stats.nmatvec >= self._krylov_nmax*stats.nonlin_iters:
                break

            # Continue a stalled solve from where it stopped
            if stats.stalled:
                # Give up after a stall on a retry
                if i > 0:
                    break

                guess_fn = warm_guess_fn
            # Restart a diverged solve from the initial guess
            else:
                guess_fn = initial_guess_fn

            krylov_rtol *= self._retry_tol_fact
            safeguard = False

        dt = time.perf_counter() - self.backend.kernel_setup_wtime - tstart
        wtime = scal_coll(comm.Allreduce, dt, op=mpi.MAX)

        # Inform the tolerance controller unless a build inflated the time
        built = not pc_built_before and self._precond_computed
        if not built:
            self._tol_controller.update(wtime, gamma_dt, stats.converged,
                                        niters)

        return stats._replace(
            nonlin_iters=niters, precond_apps=nprecond, nmatvec=nmatvec,
            nretries=i, inner_tol=krylov_rtol, probe=int(probing)
        )
