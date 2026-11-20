import numpy as np
import pytest

from pyfr.integrators.implicit.base import SolveStats
from pyfr.integrators.implicit.steppers import (Kvaerno43Stepper,
                                                TRBDF2Stepper)


# Linear test problem: du/dt = L u + g(t)
L = np.array([[-1.0, 0.5], [-0.5, -2.0]])


def g(t):
    return np.array([np.sin(3*t), np.cos(2*t)])


def f(t, u):
    return L @ u + g(t)


def make_stepper(stepcls):
    class NumPySDIRK(stepcls):
        controller_needs_errest = True
        comp_accum = False

        def __init__(self, u0):
            self._nstages = len(self.b)
            self.c = [sum(row) for row in self.A]
            self._err_coeffs = [b - bh for b, bh in zip(self.b, self.bhat)]
            self._guess_norms = {}
            self._fsal = self.A[0][0] == 0 and self.A[-1] == self.b
            self._fsal_reg = None
            self._stage_stats = []

            # Solution, stage, error, scratch, and stage flux registers
            n = self._nstages
            self._regs = np.zeros((n + 4, len(u0)))
            self._r_u, self._r_s, self._r_err, self._tmp = range(4)
            self._r_f = list(range(4, n + 4))
            self._regs[self._r_u] = u0
            self.idxcurr = self._r_u

        def _addv(self, consts, regidxs, in_scale=(), in_scale_idxs=(),
                  out_scale=()):
            x, (a0, *an), (r0, *rn) = self._regs, consts, regidxs
            acc = sum(c*x[r] for c, r in zip(an, rn))
            x[r0] = acc if a0 == 0 else a0*x[r0] + acc

        def _rhs(self, t, uin, fout):
            self._regs[fout] = f(t, self._regs[uin])

        def _stage_solve(self, i, t, u_reg, f_reg, residual_fn,
                         initial_guess_fn, gamma_dt):
            x = self._regs

            # Residual at u = f = 0 gives minus the known terms
            x[u_reg] = x[f_reg] = 0
            residual_fn(u_reg, f_reg, self._tmp)

            # Solve u - gamma_dt*(L u + g) = known exactly
            I = np.eye(len(L))
            x[u_reg] = np.linalg.solve(I - gamma_dt*L,
                                       gamma_dt*g(t) - x[self._tmp])
            x[f_reg] = f(t, x[u_reg])

            return SolveStats(nonlin_iters=1, precond_apps=0, init_resid=1,
                              final_resid=0, converged=1, stalled=0)

        def take_step(self, t, dt, accept=True):
            icurr, iprev, ierr = self.step(t, dt)
            if accept:
                self.idxcurr = icurr
            else:
                self.idxcurr = iprev

            return self._regs[icurr].copy(), self._regs[ierr].copy()

    return NumPySDIRK


def ref_step(stepcls, t, u, dt):
    A, b, bhat = stepcls.A, stepcls.b, stepcls.bhat
    c = [sum(row) for row in A]
    I = np.eye(len(u))

    # Directly compute the stage fluxes of the SDIRK scheme
    fs = []
    for i, Ai in enumerate(A):
        ti = t + c[i]*dt
        rhs = u + dt*sum(Ai[j]*fs[j] for j in range(i)) + dt*Ai[i]*g(ti)
        ui = np.linalg.solve(I - dt*Ai[i]*L, rhs)
        fs.append(f(ti, ui))

    unew = u + dt*sum(bi*fi for bi, fi in zip(b, fs))
    err = dt*sum((bi - bhi)*fi for bi, bhi, fi in zip(b, bhat, fs))

    return unew, err


steppers = [TRBDF2Stepper, Kvaerno43Stepper]


@pytest.mark.parametrize('stepcls', steppers)
def test_sdirk_error_estimate(stepcls):
    u0, dt = np.array([1.0, -0.5]), 0.1
    s = make_stepper(stepcls)(u0)

    # Take several steps so that FSAL reuse is exercised
    t, u = 0.0, u0
    for _ in range(3):
        unew, err = s.take_step(t, dt)
        uref, eref = ref_step(stepcls, t, u, dt)

        assert np.allclose(unew, uref, rtol=1e-12, atol=1e-14)
        assert np.allclose(err, eref, rtol=1e-10, atol=1e-14)

        t, u = t + dt, unew


@pytest.mark.parametrize('stepcls', steppers)
def test_sdirk_rejected_step(stepcls):
    u0, dt = np.array([1.0, -0.5]), 0.1

    # Accept a step, reject a large one, then retry with a smaller one
    s = make_stepper(stepcls)(u0)
    u1, _ = s.take_step(0.0, dt)
    s.take_step(dt, 4*dt, accept=False)
    u2, err2 = s.take_step(dt, dt/2)

    # Compare against the same sequence without the rejected step
    uref, eref = ref_step(stepcls, dt, u1, dt/2)

    assert np.allclose(u2, uref, rtol=1e-12, atol=1e-14)
    assert np.allclose(err2, eref, rtol=1e-10, atol=1e-14)
