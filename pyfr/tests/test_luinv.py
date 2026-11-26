"""Tests of the batched LU inverses of the block-Jacobi preconditioner.

Each implementation (the Mako kernels and, on HIP, the hand written ones)
inverts I - scale*J for Jacobians J of several kinds.  A sample of the
inverses is checked against np.linalg.inv in float64, with
error bounds scaled to the precision.  A quick set of cases runs by
default; the options in conftest.py select more:

  --luinv-full                the whole validation matrix
  --luinv-bench               benchmark the HIP kernels against Mako, and
                              run nothing else
  --luinv-bench-report FILE   also write the benchmark to FILE
"""
import gc
import itertools as it

import numpy as np
import pytest

from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.integrators.implicit.luinv import LUInverse, get_lu_inverse
from pyfr.nputil import bf16_to_f32, bfloat16, f32_to_bf16, is_bf16


# Stored inverse and staged Jacobian dtypes of each precond-precision, as
# the block-Jacobi preconditioner picks them
PRECISIONS = {
    'double': (np.float64, np.float64),
    'single': (np.float32, np.float32),
    'half': (np.float16, np.float32),
    'bf16': (bfloat16, bfloat16),
}

# The implementations: Mako on each backend, and hand written on HIP
IMPLS = [('openmp', 'mako'), ('hip', 'mako'), ('hip', 'hip')]
HIP_IMPLS = ['mako', 'hip']

# The scale gamma*dt of the Jacobians
SCALE = 0.05


# Inputs: Jacobians J for which I - scale*J is a matrix of a given kind

def _jacobian(rng, k, n):
    # Shifted Gaussian, like I - gamma*dt*J with a moderate spectral radius
    return np.eye(n) - 0.9/np.sqrt(n)*rng.standard_normal((k, n, n))


def _nopivot(rng, k, n):
    # Strictly column diagonally dominant so partial pivoting never swaps
    a = rng.standard_normal((k, n, n))
    i = np.arange(n)
    a[:, i, i] = 0
    a[:, i, i] = 1.1*np.abs(a).sum(axis=1)*rng.choice([-1, 1], (k, n))
    return a


def _pivot(rng, k, n):
    # Rows of a pivot free matrix shuffled so nearly every column pivots
    a = _nopivot(rng, k, n)
    for m in a:
        m[:] = m[rng.permutation(n)]
    return a


def _dense(rng, k, n):
    # Unstructured Gaussian matrices; condition numbers of O(n) and above
    return rng.standard_normal((k, n, n)) / np.sqrt(n)


def _illcond(rng, k, n, cond=1e6, npool=4):
    # Few orthogonally similar matrices of the given condition number, made
    # distinct per element through a scale factor; the singular values are
    # centred on one so the inverses stay in range of half precision
    s = np.logspace(np.log10(cond) / 2, -np.log10(cond) / 2, n)
    pool = []
    for _ in range(min(k, npool)):
        u, _ = np.linalg.qr(rng.standard_normal((n, n)))
        v, _ = np.linalg.qr(rng.standard_normal((n, n)))
        pool.append((u*s) @ v.T)

    f = rng.uniform(0.5, 2, k)
    return np.array([fi*pool[i % len(pool)] for i, fi in enumerate(f)])


KINDS = {'jacobian': _jacobian, 'nopivot': _nopivot, 'pivot': _pivot,
         'dense': _dense, 'illcond': _illcond}


def make_jacobians(kind, k, n, seed=0):
    a = KINDS[kind](np.random.default_rng(seed), k, n)
    return (np.eye(n) - a) / SCALE


# The reference and the error metric

def roundoffs(precision):
    pcdtype, wdtype = PRECISIONS[precision]

    # The factorisation is in the staging precision, with bf16 in single
    fdtype = np.float32 if is_bf16(wdtype) else wdtype
    uf = float(np.finfo(fdtype).eps) / 2

    # Relative and absolute (underflow) rounding errors of the stored inverse
    if is_bf16(pcdtype):
        uo, eta = 2.0**-8, 2.0**-134
    else:
        uo = float(np.finfo(pcdtype).eps) / 2
        eta = float(np.finfo(pcdtype).smallest_subnormal) / 2

    return uf, uo, eta


def _infnorm(m):
    return np.abs(m).sum(axis=-1).max(axis=-1)


def inverse_errors(a, x, precision):
    """Per matrix errors of the inverses x of a, relative to first order
    error bounds, so that values of at most one are acceptable whatever
    the precision:

      res  = ||a x - I|| / (||a|| ||x|| (n uf + uo + n eta / ||x||))
      ferr = ||x - inv(a)|| / (||inv(a)|| (kappa n uf + uo
                                           + n eta / ||inv(a)||))

    where uf and uo are the unit roundoffs of the factorisation and of the
    stored inverse, eta the absolute rounding error of its subnormals, and
    kappa the condition number of a, all in the infinity norm.  The eta
    terms matter for half precision, where small entries of an inverse are
    subnormal.
    """
    n = a.shape[-1]
    uf, uo, eta = roundoffs(precision)

    xref = np.linalg.inv(a)
    na, nx, nxref = _infnorm(a), _infnorm(x), _infnorm(xref)
    kappa = na*nxref

    res = _infnorm(a @ x - np.eye(n)) / (na*nx)
    ferr = _infnorm(x - xref) / nxref

    return {
        'finite': np.isfinite(x).all(axis=(-2, -1)),
        'kappa': kappa,
        'res': res / (n*uf + uo + n*eta/nx),
        'ferr': ferr / (kappa*n*uf + uo + n*eta/nxref),
    }


def passes(a, x, precision):
    err = inverse_errors(a, x, precision)
    return (err['finite'].all() and err['res'].max() <= 1 and
            err['ferr'].max() <= 1)

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


def to_float64(ary):
    if is_bf16(ary.dtype):
        return bf16_to_f32(ary).reshape(ary.shape).astype(np.float64)
    else:
        return ary.astype(np.float64)


def from_float64(ary, dtype):
    if is_bf16(dtype):
        return f32_to_bf16(ary).reshape(ary.shape).view(dtype)
    else:
        return ary.astype(dtype)


def make_colours(neles, ncolours, seed=0):
    # Scatter the elements over the colours as a mesh colouring would
    rng = np.random.default_rng(seed)
    cols = rng.permutation(np.arange(neles) % ncolours)

    return {c: np.flatnonzero(cols == c) for c in range(ncolours)}


class Problem:
    def __init__(self, bname, n, colours, precision):
        # Single precision inverses on a single precision backend, and the
        # others on a double precision one
        bprec = 'single' if precision == 'single' else 'double'
        self.backend = backend = _backend(bname, bprec)

        self.n, self.precision, self.colours = n, precision, colours
        self.pcdtype, self.wdtype = PRECISIONS[precision]
        self.neles = 1 + max(ix.max() for ix in colours.values())
        nemax = max(len(ix) for ix in colours.values())

        # Staged Jacobians of one colour and the inverses of all elements
        self.jstage = backend.matrix((n, n, nemax), dtype=self.wdtype,
                                     extent='luinv', tags={'align'})
        self.inv = backend.tiled_matrix(block_shape=(n, n), nmats=self.neles,
                                        dtype=self.pcdtype)

        self.luinv = get_lu_inverse(backend, self.jstage, self.inv, colours,
                                    'luinv')
        backend.commit()

    def stage(self, colour, jac):
        # Lay the Jacobians of the colour out as the extract kernel does
        ary = np.zeros(self.jstage.ioshape)
        ary[..., :len(jac)] = jac.transpose(1, 2, 0)

        ary = from_float64(ary, self.wdtype)
        self.jstage.set(ary)

        # Return the Jacobians as staged, after any rounding
        return to_float64(ary[..., :len(jac)]).transpose(2, 0, 1)

    def invert(self, colour, scale=SCALE):
        self.luinv.invert(colour, scale)

    def read(self):
        mat, backend = self.inv, self.backend

        # Copy out the tiles of every matrix in its padded row-major order
        if backend.name == 'hip':
            backend.wait()

            buf = np.empty(mat.nbytes // mat.itemsize, dtype=mat.dtype)
            backend.hip.memcpy(buf, mat.data, mat.nbytes)

            buf = buf.reshape(mat.nmats, mat.ntiles_r, mat.ntiles_c,
                              mat.trows, mat.tcols)
            buf = buf.transpose(0, 1, 3, 2, 4)
        else:
            buf = mat.data.transpose(0, 5, 1, 3, 2, 4)

        buf = buf.reshape(-1, mat.padr, mat.padc)
        buf = buf[:mat.nmats, :mat.nrow, :mat.ncol]
        return to_float64(np.ascontiguousarray(buf).view(mat.dtype))

    def fill(self, value):
        # Overwrite every stored inverse
        ary = np.full((self.neles, self.n, self.n), value)
        self.inv._set(from_float64(ary, self.pcdtype))
        self.backend.wait()

    def stage_and_invert(self, kind='jacobian', colours=None, nsample=32,
                         seed=0):
        """Stage and invert the colours, returning a sample of the elements
        and their matrices I - scale*J as staged."""
        rng = np.random.default_rng(seed)
        eidx, a = [], []
        for c in self.colours if colours is None else colours:
            ix = self.colours[c]
            jac = make_jacobians(kind, len(ix), self.n, seed=(seed, c))
            jac = self.stage(c, jac)
            self.invert(c)

            # Always the first and last element of the colour
            pick = np.union1d([0, len(ix) - 1],
                              rng.permutation(len(ix))[:nsample])
            eidx.append(ix[pick])
            a.append(np.eye(self.n) - SCALE*jac[pick])

        return np.concatenate(eidx), np.concatenate(a)


@pytest.fixture
def make_problem(monkeypatch):
    # Pick the implementation as a user would, through the environment
    def make(bname, impl, n, colours, precision='double'):
        monkeypatch.setenv('PYFR_HIP_LU_INVERSE', impl)
        return Problem(bname, n, colours, precision)

    yield make

    # Problems hold large workspaces so free them between tests
    gc.collect()


# The checks of the metric itself

def _exact(kind, n=64, k=4):
    a = np.eye(n) - SCALE*make_jacobians(kind, k, n)
    return a, np.linalg.inv(a)


@pytest.mark.parametrize('precision', PRECISIONS)
def test_metric_passes(precision):
    # Rounding an exact inverse to the stored precision is acceptable
    for kind in KINDS:
        a, x = _exact(kind)
        x = to_float64(from_float64(x, PRECISIONS[precision][0]))
        assert passes(a, x, precision)


def test_metric_passes_subnormal():
    # Also when entries below the smallest normal half become subnormal
    a, x = _exact('nopivot', n=320)
    assert np.median(np.abs(x)) < np.finfo(np.float16).tiny
    assert passes(a, to_float64(x.astype(np.float16)), 'half')


def test_metric_fails():
    a, x = _exact('jacobian', n=320)

    # A single wrong entry well above roundoff
    for rel in (1e-9, 1e-3):
        y = x.copy()
        y[2, 5, 7] += rel*np.abs(x[2]).max()
        assert not passes(a, y, 'double')

    # Inverses written to the wrong elements, or not finite
    assert not passes(a, x[[1, 0, 2, 3]], 'double')

    x[0, 0, 0] = np.nan
    assert not passes(a, x, 'double')


# The inverses of each implementation

QUICK = [
    (40, 'double', 'jacobian', 70, 3),
    *[(n, p, 'jacobian', 65, 2) for n in (100, 320) for p in PRECISIONS],
    # Either side of where the HIP policy changes the shape of the kernels
    (96, 'double', 'jacobian', 65, 1),
    (97, 'double', 'pivot', 65, 1),
    (416, 'double', 'pivot', 65, 1),
    (417, 'double', 'jacobian', 65, 1),
    (512, 'single', 'illcond', 65, 1),
    (513, 'single', 'jacobian', 65, 1),
    (175, 'double', 'pivot', 130, 1),
    (175, 'double', 'dense', 64, 1),
    (175, 'single', 'illcond', 65, 1),
    # Several workspaces of matrices
    (320, 'double', 'jacobian', 513, 1),
    (625, 'double', 'pivot', 65, 1),
]

FULL = [
    *[(n, p, k, 65, 1) for n, p, k in it.product(
        (100, 175, 320, 625), PRECISIONS, KINDS
    )],
    # The other regimes of the HIP policy
    *[(n, p, k, 65, 1) for n, p, k in it.product(
        (70, 250, 400, 500), ('double', 'single'), KINDS
    )],
    *[(175, 'double', 'jacobian', e, 1)
      for e in (1, 7, 63, 64, 65, 512, 513, 1500)],
    (320, 'double', 'jacobian', 4096, 8),
]


def _cases():
    # The implementations of each case in turn
    for cases, marks in [(QUICK, ()), (FULL, pytest.mark.full)]:
        for case, (bname, impl) in it.product(cases, IMPLS):
            n, prec = case[:2]

            # The CPU only for the small cases in double and single
            if bname == 'openmp' and (marks or n > 100 or
                                      prec not in ('double', 'single')):
                continue

            yield pytest.param(
                bname, impl, *case, marks=marks,
                id='{}-{}-n{}-{}-{}-e{}x{}'.format(bname, impl, *case)
            )


@pytest.mark.parametrize('bname,impl,n,prec,kind,neles,ncolours',
                         list(_cases()))
def test_inverse(make_problem, bname, impl, n, prec, kind, neles, ncolours):
    prob = make_problem(bname, impl, n, make_colours(neles, ncolours), prec)
    eidx, a = prob.stage_and_invert(kind)

    assert passes(a, prob.read()[eidx], prec)


@pytest.mark.parametrize('n', [100, 320])
def test_matches_mako(make_problem, n):
    colours = make_colours(130, 2)

    xs = []
    for impl in HIP_IMPLS:
        prob = make_problem('hip', impl, n, colours)
        eidx, a = prob.stage_and_invert()
        xs.append(prob.read()[eidx])

    # Agreement to within the error bounds of the inverses themselves
    xm, xh = xs
    kappa = inverse_errors(a, xm, 'double')['kappa']
    rel = _infnorm(xh - xm) / _infnorm(xm)

    assert np.all(rel <= kappa*n*np.finfo(np.float64).eps)


@pytest.mark.parametrize('impl', HIP_IMPLS)
def test_colour_isolation(make_problem, impl):
    prob = make_problem('hip', impl, 175, make_colours(200, 3))

    # Poison every stored inverse then invert a single colour
    prob.fill(np.nan)
    eidx, a = prob.stage_and_invert(colours=[1])
    x = prob.read()

    # Only the elements of that colour may have been written
    others = np.setdiff1d(np.arange(prob.neles), prob.colours[1])
    assert np.isnan(x[others]).all()
    assert passes(a, x[eidx], 'double')

    # Inverting the remaining colours must cover every element
    eidx, a = prob.stage_and_invert(colours=[0, 2])
    x = prob.read()
    assert np.isfinite(x).all()
    assert passes(a, x[eidx], 'double')


@pytest.mark.parametrize('impl', HIP_IMPLS)
def test_deterministic(make_problem, impl):
    prob = make_problem('hip', impl, 320, make_colours(300, 1))
    prob.stage_and_invert()
    x1 = prob.read()

    prob.fill(0)
    prob.invert(0)

    assert np.array_equal(x1, prob.read())


@pytest.mark.parametrize('impl', HIP_IMPLS)
def test_rescale(make_problem, impl):
    # Inverting the staged Jacobians again with another scale must not
    # reuse any state from the first inversion
    prob = make_problem('hip', impl, 320, make_colours(100, 1))
    prob.stage_and_invert()
    prob.invert(0, 2*SCALE)

    jac = to_float64(prob.jstage.get()).transpose(2, 0, 1)
    a = np.eye(prob.n) - 2*SCALE*jac

    assert passes(a, prob.read(), 'double')


def test_select(make_problem, monkeypatch):
    from pyfr.backends.hip.luinv import HIPLUInverse

    def impl(env='', cfg=None):
        backend = _backend('hip', 'double')

        # Configure the backend through a copy of its config
        bcfg = Inifile(backend.cfg.tostr())
        if cfg:
            bcfg.set('backend-hip', 'lu-inverse', cfg)
        monkeypatch.setattr(backend, 'cfg', bcfg)

        prob = make_problem('hip', env, 32, make_colours(64, 1))
        return type(prob.luinv)

    # Mako by default, or as configured, unless the environment overrides it
    assert impl() is LUInverse
    assert impl(cfg='hip') is HIPLUInverse
    assert impl(env='hip', cfg='mako') is HIPLUInverse
    assert impl(env='mako', cfg='hip') is LUInverse

    with pytest.raises(ValueError):
        impl(cfg='magma')

    # The hand written kernels need MFMA
    monkeypatch.setattr(HIPLUInverse, 'supported', lambda backend: False)
    with pytest.raises(ValueError):
        impl(env='hip')


# The benchmark of the HIP kernels against Mako

BENCH = list(it.product((70, 175, 320, 625), ('double', 'single')))


def _time(prob, reps=10, warmup=2):
    backend = prob.backend
    start, stop = backend.hip.create_event(), backend.hip.create_event()

    # The median time to invert the staged matrices on the device
    times = []
    for i in range(warmup + reps):
        start.record(backend._stream)
        prob.invert(0)
        stop.record(backend._stream)
        stop.synchronize()

        times.append(stop.elapsed_time(start))

    return float(np.median(times[warmup:]))


@pytest.mark.bench
@pytest.mark.parametrize('n,prec', BENCH, ids=[f'n{n}-{p}' for n, p in BENCH])
def test_bench(make_problem, request, n, prec):
    neles = 2048 if n > 512 else 4096

    times = {}
    for impl in HIP_IMPLS:
        prob = make_problem('hip', impl, n, make_colours(neles, 1), prec)
        arch = prob.backend.props['gcn_arch_name']

        # Check a few of the inverses, then time inverting them all again
        eidx, a = prob.stage_and_invert(nsample=8)
        assert passes(a, prob.read()[eidx], prec)

        times[impl] = _time(prob)

        del prob
        gc.collect()

    request.config.luinv_bench.append({
        'n': n, 'precision': prec, 'neles': neles, 'gpu': arch,
        'mako_ms': 1e3*times['mako'], 'hip_ms': 1e3*times['hip'],
        'speedup': times['mako'] / times['hip']
    })
