import numpy as np
import pytest

from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.integrators.implicit.luinv import LUInverse


_backends = {}


def _backend(name, precision):
    key = (name, precision)
    if key not in _backends:
        cfg = Inifile()
        cfg.set('backend', 'precision', precision)
        cfg.set(f'backend-{name}', 'device-id', '0')

        try:
            _backends[key] = get_backend(name, cfg)
        except Exception as e:
            _backends[key] = e

    if isinstance(_backends[key], Exception):
        pytest.skip(f'{name} backend unavailable: {_backends[key]}')

    return _backends[key]


def _read_tiled(mat):
    backend = mat.backend

    # Gather the padded row-major tiles of each matrix
    if backend.name == 'hip':
        backend.wait()

        buf = np.empty(mat.nbytes // mat.itemsize, dtype=mat.dtype)
        backend.hip.memcpy(buf, mat.data, mat.nbytes)

        buf = buf.reshape(mat.nmats, mat.ntiles_r, mat.ntiles_c, mat.trows,
                          mat.tcols)
        buf = buf.transpose(0, 1, 3, 2, 4)
    elif backend.name == 'openmp':
        buf = mat.data.transpose(0, 5, 1, 3, 2, 4)
    else:
        pytest.skip(f'No tiled readback for {backend.name}')

    buf = buf.reshape(-1, mat.padr, mat.padc)
    return buf[:mat.nmats, :mat.nrow, :mat.ncol].astype(np.float64)


@pytest.mark.parametrize('bname', ['hip', 'openmp'])
@pytest.mark.parametrize('precision', ['double', 'single'])
@pytest.mark.parametrize('n', [40, 135])
def test_luinv(bname, precision, n):
    backend = _backend(bname, precision)
    dtype = backend.fpdtype
    rng = np.random.default_rng(n)

    # Scatter the elements over three colours of unequal size
    neles, scale = 70, 0.05
    cols = rng.permutation(np.arange(neles) % 3)
    idxs = {c: np.flatnonzero(cols == c) for c in range(3)}
    nemax = max(map(len, idxs.values()))

    # Allocate the staging and inverse matrices as the preconditioner does
    jstage = backend.matrix((n, n, nemax), dtype=dtype, extent='luinv',
                            tags={'align'})
    inv = backend.tiled_matrix(block_shape=(n, n), nmats=neles, dtype=dtype)
    luinv = LUInverse(backend, jstage, inv, idxs, 'luinv')
    backend.commit()

    # Stage and invert each colour in turn
    a = np.empty((neles, n, n))
    for c, eidx in idxs.items():
        jac = rng.standard_normal((n, n, nemax)) / (scale*np.sqrt(n))
        jac = jac.astype(dtype)
        jstage.set(jac)
        luinv.invert(c, scale)
        backend.wait()

        a[eidx] = np.eye(n) - scale*jac[..., :len(eidx)].transpose(2, 0, 1)

    x = _read_tiled(inv)

    # Compare against a reference inverse
    xref = np.linalg.inv(a)
    nrm = lambda m: np.abs(m).sum(axis=-1).max(axis=-1)
    kappa = nrm(a)*nrm(xref)
    tol = 10*kappa*n*np.finfo(dtype).eps

    assert np.all(nrm(x - xref) / nrm(xref) <= tol)
    assert np.all(nrm(a @ x - np.eye(n)) / (nrm(a)*nrm(x)) <= tol)
