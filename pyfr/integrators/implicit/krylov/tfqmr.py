from pyfr.integrators.implicit.krylov.base import BaseLinearSolver
from pyfr.integrators.registers import VectorRegister


class TFQMRMixin(BaseLinearSolver):
    linear_name = 'tfqmr'

    # Stall detection window
    _stall_window = 10

    _tq = VectorRegister(n=9, rhs=False, extent='solve')

    def _krylov_solve(self, matvec, residual, out_reg, precond=None, *,
                      rtol, accumulate=True, accumulate_scale=()):
        w, y1, y2, u1, u2, v, d, x, rt = self._tq
        niters = m = 0

        # Matrix-vector product operator with optional right-preconditioning
        def op(src, dst):
            nonlocal niters
            niters += 1

            if precond:
                precond(src, self._precond_temp)
                matvec(self._precond_temp, dst)
            else:
                matvec(src, dst)

        # One QMR half-step
        def half_step(yj, uj):
            nonlocal theta, tau, eta, m, tau_min, stall
            self._add(1, w, -alpha, uj)

            # d = yj + (θ²η/α)·d; the zero coefficient sets d on entry
            self._add(theta*theta*eta/alpha, d, 1, yj)

            theta = self._norm2(w) / tau
            c = (1.0 + theta*theta)**-0.5
            tau *= theta*c
            eta = c*c*alpha

            # Overwrite x on the first update (m == 0), then accumulate
            self._add(1 if m else 0, x, eta, d)
            m += 1

            # Reset on a >=0.1% tau drop, else count toward the stall window
            if tau < 0.999*tau_min:
                tau_min, stall = tau, 0
            else:
                stall += 1

            # Return True if converged or stalled
            return tau < rtol or stall >= self._stall_window

        # If the initial residual is non-zero then start iterating
        if (r0norm := self._norm2(residual)):
            self._add(0, w, -1/r0norm, residual)
            self._add(0, y1, 1, w)

            # Fixed unit shadow vector rt = r0 (so rt·r0 = 1)
            self._add(0, rt, 1, w)

            op(y1, v)
            self._add(0, u1, 1, v)

            tau, theta, eta = 1.0, 0.0, 0.0
            tau_min, stall = 1.0, 0
            rho = self._dot(rt, w)

            while niters < self._krylov_nmax:
                # Check for breakdown
                if abs(sigma := self._dot(rt, v)) < self._breakdown_tol:
                    break

                alpha = rho / sigma

                # Take the first half-step and check for convergence or stall
                if half_step(y1, u1):
                    break

                # y2 = y1 - alpha*v; u2 = A·M⁻¹·y2
                self._add(0, y2, 1, y1, -alpha, v)
                op(y2, u2)

                # Take the second half-step and check for convergence or stall
                if half_step(y2, u2) or niters >= self._krylov_nmax:
                    break

                # Check for breakdown
                if abs(nrho := self._dot(rt, w)) < self._breakdown_tol:
                    break

                beta, rho = nrho / rho, nrho

                # y1 = w + beta*y2; v = A·M⁻¹·y1 + beta*(u2 + beta*v)
                self._add(0, y1, 1, w, beta, y2)
                op(y1, u1)
                self._add(beta*beta, v, 1, u1, beta, u2)

        # Recover M⁻¹·ξ; skip when nothing was solved (m == 0)
        if precond and m:
            precond(x, self._precond_temp)
            corr = self._precond_temp
        else:
            corr = x

        # Fold the correction into out_reg, undoing the r0norm scaling
        sidxs = (1,) if accumulate_scale else ()
        self._add(float(accumulate), out_reg, r0norm if m else 0, corr,
                  in_scale=accumulate_scale, in_scale_idxs=sidxs)

        # Each matvec applies M⁻¹ once; recovery adds one more when m > 0
        return niters, niters + (m > 0) if precond else 0
