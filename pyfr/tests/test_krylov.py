import numpy as np
import pytest

from pyfr.integrators.implicit.krylov.gmres import GMRESMixin


class NumPyGMRES(GMRESMixin):
    # Replace the backend register with plain indices into an array
    _krylov = None

    def __init__(self, n, nmax, restart, arnoldi):
        self._krylov_nmax = nmax
        self._gmres_m = min(restart, nmax) if restart else nmax
        self._breakdown_tol = 1e3*np.finfo(float).eps
        self._arnoldi = getattr(self, f'_arnoldi_{arnoldi}')

        m = self._gmres_m
        self._H = np.empty((m + 1, m))
        self._cs, self._sn = np.empty((2, m))
        self._beta = np.empty(m + 1)

        # Krylov vectors followed by the output, residual, and precond temp
        self._regs = np.zeros((m + 4, n))
        self._krylov = list(range(m + 1))
        self.out_reg, self.resid_reg, self._precond_temp = m + 1, m + 2, m + 3

    def _addv(self, consts, regidxs, in_scale=(), in_scale_idxs=(),
              out_scale=()):
        x, (a0, *an), (r0, *rn) = self._regs, consts, regidxs
        acc = sum(c*x[r] for c, r in zip(an, rn))

        # Mirror axnpby: a zero first coefficient overwrites the output
        x[r0] = acc if a0 == 0 else a0*x[r0] + acc

    def _norm2(self, r, *, weights=(), norm_gndofs=False):
        return float(np.linalg.norm(self._regs[r]))

    def _dot(self, a, b):
        return float(self._regs[a] @ self._regs[b])

    def _multidot(self, a, *bn):
        return self._regs[list(bn)] @ self._regs[a]

    def solve(self, A, b, rtol, Minv=None):
        x = self._regs

        def matvec(v, result, vnorm=None):
            x[result] = A @ x[v]

        def precond(v, result):
            x[result] = Minv @ x[v]

        # PyFR passes the negated right hand side
        x[self.resid_reg] = -b
        self._krylov_solve(matvec, self.resid_reg, self.out_reg,
                           precond if Minv is not None else None,
                           rtol=rtol, accumulate=False)

        return x[self.out_reg].copy()


def _test_system(n=50, seed=42):
    rng = np.random.default_rng(seed)

    # Well-conditioned non-symmetric matrix with a non-uniform diagonal
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    S = rng.standard_normal((n, n))
    A = Q @ np.diag(np.linspace(1, 8, n)) @ Q.T + 0.1*(S - S.T)
    A += np.diag(rng.uniform(0, 4, n))

    return A, rng.standard_normal(n)


@pytest.mark.parametrize('precond', [False, True])
@pytest.mark.parametrize('arnoldi', ['cgs', 'mgs'])
@pytest.mark.parametrize('restart', [0, 2, 4, 8])
def test_gmres_restart(restart, arnoldi, precond):
    A, b = _test_system()
    Minv = np.diag(1 / np.diag(A)) if precond else None
    rtol = 1e-10

    solver = NumPyGMRES(len(b), nmax=300 if restart else 30, restart=restart,
                        arnoldi=arnoldi)
    x = solver.solve(A, b, rtol, Minv)

    # Check the true residual rather than the solver's own estimate
    assert np.linalg.norm(b - A @ x) / np.linalg.norm(b) < 10*rtol
