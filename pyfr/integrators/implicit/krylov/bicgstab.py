from pyfr.integrators.implicit.krylov.base import BaseLinearSolver
from pyfr.integrators.registers import VectorRegister


class BiCGStabMixin(BaseLinearSolver):
    linear_name = 'bicgstab'
    _bcg = VectorRegister(n=6, rhs=False, extent='solve')

    def _krylov_solve(self, matvec, x_reg, precond=None, *, rtol):
        r, rhat, p, v, s, t = self._bcg
        niters, xset, relerr = 0, False, 0.0

        # Matrix-vector product operator with optional right-preconditioning
        def op(src, dst):
            nonlocal niters
            niters += 1

            if precond:
                precond(src, self._precond_temp)
                matvec(self._precond_temp, dst)
                return self._precond_temp
            else:
                matvec(src, dst)
                return src

        # So long as the residual is non-zero, iterate
        if (r0norm := self._norm2(x_reg)):
            self._add(0, r, -1/r0norm, x_reg)
            self._add(0, rhat, 1, r)
            relerr = 1.0

            while niters < self._krylov_nmax:
                # Check for breakdown
                if abs(nrho := self._dot(rhat, r)) < self._breakdown_tol:
                    break

                # On the first iteration take p = r
                if niters == 0:
                    self._add(0, p, 1, r)
                # On subsequent iterations, update p = r + beta*(p - omega*v)
                else:
                    bw = alpha*(nrho / rho)
                    self._add(bw / omega, p, -bw, v, 1, r)

                # v = A·M⁻¹·p; phat = M⁻¹·p
                phat = op(p, v)

                if abs(denom := self._dot(rhat, v)) < self._breakdown_tol:
                    break

                # s = r - alpha*v; x += alpha*phat (overwrite on first write)
                alpha = nrho / denom
                self._add(0, s, 1, r, -alpha, v)
                self._add(1 if xset else 0, x_reg, alpha, phat)
                xset = True

                # Half-step convergence saves the second matvec
                relerr = self._norm2(s)
                if relerr < rtol or niters >= self._krylov_nmax:
                    break

                # t = A·M⁻¹·s; shat = M⁻¹·s
                shat = op(s, t)

                # omega = (t,s)/(t,t) minimises ‖s - omega*t‖; break on t ≈ 0
                ts, tt = self._multidot(t, s, t)
                if abs(tt) < self._breakdown_tol:
                    break

                # x += omega*shat; r = s - omega*t
                omega = ts / tt
                self._add(1, x_reg, omega, shat)
                self._add(0, r, 1, s, -omega, t)
                rho = nrho

                # Converged, or omega ≈ 0 (next p update divides by omega)
                relerr = self._norm2(r)
                if relerr < rtol or abs(omega) < self._breakdown_tol:
                    break

        # Leave the correction in x_reg, undoing the r0norm scaling
        self._add(r0norm if xset else 0, x_reg)

        return niters, niters if precond else 0, relerr
