import numpy as np

from pyfr.integrators.implicit.krylov import BaseKrylovSolver
from pyfr.integrators.implicit.tolerance import get_krylov_tol_controller
from pyfr.integrators.registers import DynamicVectorRegister


class GMRESMixin(BaseKrylovSolver):
    krylov_name = 'gmres'
    _krylov = DynamicVectorRegister(rhs=False)

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        sect = 'solver-time-integrator'

        self._gmres_nmax = cfg.getint(sect, 'krylov-max-iter', 10)

        # Arnoldi method
        match cfg.get(sect, 'gmres-arnoldi', 'cgs').lower():
            case 'cgs':
                self._arnoldi = self._arnoldi_cgs
            case 'mgs':
                self._arnoldi = self._arnoldi_mgs
            case _:
                raise ValueError('Invalid Arnoldi method: must be mgs or cgs')

        # Size the storage for the Krylov vectors
        self._size_register(self._krylov, self._gmres_nmax + 1)

        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        # Tolerance controller; created after super().__init__ so the
        # serialiser is available for the GP controller to register state
        self._tol_controller = get_krylov_tol_controller(cfg, self.serialiser,
                                                         initsoln)

        # Allocate storage for the Arnoldi process
        self._H = np.empty((self._gmres_nmax + 1, self._gmres_nmax))
        self._cs, self._sn = np.empty((2, self._gmres_nmax))
        self._beta = np.empty(self._gmres_nmax + 1)

    def _reset_gmres_arrays(self):
        self._H.fill(0)
        self._cs.fill(0)
        self._sn.fill(0)
        self._beta.fill(0)

    def _arnoldi_cgs(self, w_reg, v, H, j):
        H[:j + 1, j] = h = self._multidot(w_reg, *v[:j + 1])

        self._addv([1] + (-h).tolist(), [w_reg] + v[:j + 1])

    def _arnoldi_mgs(self, w_reg, v, H, j):
        for i, v_i in enumerate(v[:j + 1]):
            H[i, j] = h_ij = self._dot(w_reg, v_i)
            self._add(1, w_reg, -h_ij, v_i)

    def _compute_givens(self, h_jj, h_jp1_j):
        if abs(h_jp1_j) < self._breakdown_tol:
            return 1, 0
        else:
            denom = np.hypot(h_jj, h_jp1_j)
            return h_jj / denom, h_jp1_j / denom

    def _apply_givens(self, H, beta, cs, sn, j):
        # Apply previous Givens rotations to column j
        for i, (c, s) in enumerate(zip(cs[:j], sn[:j])):
            h_ij, h_ip1_j = H[i:i + 2, j]
            H[i:i + 2, j] = c*h_ij + s*h_ip1_j, -s*h_ij + c*h_ip1_j

        # Compute and apply Givens rotation to zero out H[j+1, j]
        cs[j], sn[j] = self._compute_givens(H[j, j], H[j + 1, j])
        H[j:j + 2, j] = cs[j]*H[j, j] + sn[j]*H[j + 1, j], 0

        # Also apply to beta
        beta[j:j + 2] = cs[j]*beta[j], -sn[j]*beta[j]

    def _krylov_solve(self, matvec, residual, out_reg, precond_apply=None,
                      accumulate=True, accumulate_scale=()):
        v = self._krylov

        self._reset_gmres_arrays()

        # Compute initial residual norm
        self._beta[0] = r0_norm = self._norm2(residual)

        # Normalize residual to get first Krylov vector
        self._add(0, v[0], -1/r0_norm, residual)

        # Arnoldi process with incremental Givens rotations
        for j, w_reg in enumerate(v[1:]):
            # Right preconditioning: w = A * M^{-1} * v[j]
            if precond_apply:
                precond_apply(v[j], self._precond_temp)
                matvec(self._precond_temp, w_reg)
            else:
                matvec(v[j], w_reg)

            # Arnoldi orthogonalization
            self._arnoldi(w_reg, v, self._H, j)

            # Compute h_{j+1,j} = ||w||
            self._H[j + 1, j] = h_jp1_j = self._norm2(w_reg)

            # Apply Givens rotations
            self._apply_givens(self._H, self._beta, self._cs, self._sn, j)

            # Check for convergence
            err = abs(self._beta[j + 1]) / r0_norm
            if err < self._krylov_rtol:
                break

            # Check for breakdown
            if h_jp1_j < self._breakdown_tol:
                break

            # Normalize to get v_{j+1} = w / h_{j+1,j}
            if j < self._gmres_nmax - 1:
                self._add(1/h_jp1_j, v[j + 1])

        # Backward substitution to solve for y
        y = np.linalg.solve(self._H[:j + 1, :j + 1], self._beta[:j + 1])

        # Compute solution update
        if precond_apply:
            self._addv([0, *y.tolist()], [self._precond_temp, *v[:j + 1]])
            precond_apply(self._precond_temp, v[0])
            self._add(int(accumulate), out_reg, 1, v[0],
                      in_scale=accumulate_scale,
                      in_scale_idxs=(1,) if accumulate_scale else ())
        else:
            sidxs = tuple(range(1, j + 2)) if accumulate_scale else ()
            self._addv([int(accumulate), *y.tolist()], [out_reg, *v[:j + 1]],
                       in_scale=accumulate_scale, in_scale_idxs=sidxs)

        return j + 1, (j + 2) if precond_apply else 0
