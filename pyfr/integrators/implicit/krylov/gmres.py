import numpy as np

from pyfr.integrators.implicit.krylov.base import BaseLinearSolver
from pyfr.integrators.registers import DynamicVectorRegister


class GMRESMixin(BaseLinearSolver):
    linear_name = 'gmres'
    _krylov = DynamicVectorRegister(rhs=False, extent='solve')

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        # Obtain the matvec budget
        nmax = cfg.getint('solver-time-integrator', 'linear-max-iter', 10)

        # Restart size; 0 means no restart (single cycle of nmax)
        m = cfg.getint('solver-gmres', 'restart', 0)
        self._gmres_m = min(m, nmax) if m else nmax

        # Arnoldi method
        match cfg.get('solver-gmres', 'arnoldi', 'cgs').lower():
            case 'cgs':
                self._arnoldi = self._arnoldi_cgs
            case 'mgs':
                self._arnoldi = self._arnoldi_mgs
            case _:
                raise ValueError('Invalid Arnoldi method: must be mgs or cgs')

        # Size the storage for the Krylov vectors
        self._size_register(self._krylov, self._gmres_m + 1)

        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        # Allocate storage for the Arnoldi process
        self._H = np.empty((self._gmres_m + 1, self._gmres_m))
        self._cs, self._sn = np.empty((2, self._gmres_m))
        self._beta = np.empty(self._gmres_m + 1)

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

    def _krylov_solve(self, matvec, residual, out_reg, precond=None, *,
                      rtol, accumulate=True, accumulate_scale=()):
        v, m, nmax = self._krylov, self._gmres_m, self._krylov_nmax

        niters = ncycles = 0

        # Compute initial residual norm and first Krylov vector
        r0norm = self._norm2(residual)
        self._add(0, v[0], -1/r0norm, residual)
        rnorm = r0norm

        while niters < nmax:
            self._reset_gmres_arrays()
            self._beta[0] = rnorm

            budget = min(m, nmax - niters)

            # Arnoldi process with incremental Givens rotations
            for j, w_reg in enumerate(v[1:budget + 1]):
                # Right preconditioning: w = A·M⁻¹·v[j]
                if precond:
                    precond(v[j], self._precond_temp)
                    matvec(self._precond_temp, w_reg)
                else:
                    # v[j] is a unit Arnoldi vector, so ‖v[j]‖ = 1
                    matvec(v[j], w_reg, 1.0)

                # Arnoldi orthogonalization
                self._arnoldi(w_reg, v, self._H, j)

                # Compute h_{j+1,j} = ‖w‖
                self._H[j + 1, j] = h_jp1_j = self._norm2(w_reg)

                # Apply Givens rotations
                self._apply_givens(self._H, self._beta, self._cs, self._sn, j)

                # Check for convergence or breakdown
                err = abs(self._beta[j + 1]) / r0norm
                if err < rtol or h_jp1_j < self._breakdown_tol:
                    break

                # Normalize to get v_{j+1} = w / h_{j+1,j}
                if j < budget - 1:
                    self._add(1/h_jp1_j, v[j + 1])

            niters += j + 1
            ncycles += 1

            # Backward substitution to solve for y
            y = np.linalg.solve(self._H[:j + 1, :j + 1], self._beta[:j + 1])

            # Solution update; first cycle honours accumulate, rest add
            first = niters == j + 1
            acc = float(accumulate) if first else 1.0

            if precond:
                self._addv([0, *y.tolist()], [self._precond_temp, *v[:j + 1]])
                precond(self._precond_temp, v[0])
                self._add(acc, out_reg, 1, v[0], in_scale=accumulate_scale,
                          in_scale_idxs=(1,) if accumulate_scale else ())
            else:
                sidxs = tuple(range(1, j + 2)) if accumulate_scale else ()
                self._addv([acc, *y.tolist()], [out_reg, *v[:j + 1]],
                           in_scale=accumulate_scale, in_scale_idxs=sidxs)

            if err < rtol or h_jp1_j < self._breakdown_tol or niters >= nmax:
                break

            # Restart: recover residual via r_m = beta[j+1]*v_{j+1}/h_{j+1,j}
            rnorm = abs(self._beta[j + 1])
            s = np.copysign(1 / h_jp1_j, self._beta[j + 1])
            self._add(0, v[0], s, v[j + 1])

        # Each matvec applies M⁻¹ once; recovery adds one more per cycle
        return niters, niters + ncycles if precond else 0
