import math

import numpy as np

from pyfr.integrators.implicit.nonlinear.base import (BaseNonlinearSolver,
                                                      NonlinearDivergenceError)
from pyfr.integrators.registers import DynamicVectorRegister, VectorRegister


class AndersonSolver(BaseNonlinearSolver):
    nonlinear_name = 'anderson'
    requires_linear_solver = False

    # Anderson work/history registers reuse the shared 'solve' extent scratch
    _aa = VectorRegister(n=3, rhs=False, extent='solve')
    _aa_dF = DynamicVectorRegister(rhs=False, extent='solve')
    _aa_dX = DynamicVectorRegister(rhs=False, extent='solve')

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        # Anderson history depth; 0 reduces to preconditioned Picard
        self._aa_depth = cfg.getint('solver-anderson', 'depth', 5)

        # Size the depth-dependent history registers
        self._size_register(self._aa_dF, self._aa_depth)
        self._size_register(self._aa_dX, self._aa_depth)

        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        # Anderson needs a preconditioner to make the fixed point contractive
        if not self._preconditioner.active:
            raise ValueError('nonlinear-solver = anderson requires a '
                             'preconditioner')

    def _solve(self, t, u_reg, f_reg, residual_fn, initial_guess_fn, gamma_dt):
        g, fprev, uprev = self._aa
        dF, dX = self._aa_dF, self._aa_dX

        # Scaled preconditioned increment g̃ = S⁻¹ M⁻¹ F (weighted residual)
        def precond(in_reg, out_reg):
            self._apply_precond(in_reg, out_reg, out_scale=self._inv_scales)

        rnorm = rnorm0 = initial_guess_fn(u_reg)
        if not math.isfinite(rnorm):
            raise NonlinearDivergenceError('Non-finite residual')

        self._compute_fd_eps(u_reg)

        nrhs, nprec = 1, 0
        tol = max(rnorm*self._nl_rtol, self._nl_atol)

        # Build the block preconditioner once and freeze it over the stage
        self._compute_precond(t, u_reg, gamma_dt, self._rhs, f_reg,
                              self._nl_temp, eps_scales=self._scales)

        for k in range(self._nl_maxiter):
            if not math.isfinite(rnorm):
                raise NonlinearDivergenceError('Non-finite residual')
            if k > 0 and rnorm < tol:
                break

            # Preconditioned step g = M̃⁻¹·F; the increment is f_k = -g
            precond(self._nl_resid, g)
            nprec += 1

            mk = min(k, self._aa_depth)

            # Build new history columns Δf, Δx from the previous iterate
            if k > 0 and self._aa_depth:
                slot = (k - 1) % self._aa_depth
                self._add(0, dF[slot], -1, g, -1, fprev)
                self._add(0, dX[slot], 1, u_reg, -1, uprev,
                          out_scale=self._inv_scales)

            # Stash the current increment and iterate for the next step
            self._add(0, fprev, -1, g)
            self._add(0, uprev, 1, u_reg)

            if mk:
                cols, xcols = dF[:mk], dX[:mk]

                # γ = argmin ‖f_k - ΔF·γ‖ via the (small) normal equations
                b = -self._multidot(g, *cols)
                a = np.array([self._multidot(c, *cols) for c in cols])
                gamma = np.linalg.lstsq(a, b, rcond=None)[0]
                ng = (-gamma).tolist()
            else:
                cols = xcols = ng = []

            # Scaled Anderson increment Δũ = f̃_k - Σ γ_i (Δx̃_i + Δf̃_i)
            if self._linesearch:
                self._addv([0, -1, *ng, *ng],
                           [self._nl_delta, g, *xcols, *cols])

                # Backtracking line search globalises the Anderson step
                rnorm = self._line_search(t, u_reg, f_reg, self._nl_delta,
                                          residual_fn, rnorm)
                nrhs += 2
            else:
                # u = u_k + f_k - Σ γ_i (Δx_i + Δf_i), unscaling by S
                regs = [u_reg, g, *xcols, *cols]
                self._addv([1, -1, *ng, *ng], regs, in_scale=self._scales,
                           in_scale_idxs=tuple(range(1, len(regs))))
                rnorm = self._residual_norm(t, u_reg, f_reg, residual_fn)
                nrhs += 1

        return (k + 1, nrhs, nprec, rnorm0, rnorm, self._nl_rtol)
