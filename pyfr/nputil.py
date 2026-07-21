import contextlib
import ctypes as ct
import functools as ft
from math import erf, prod
from pathlib import Path
import sys

import numpy as np


def block_diag(arrs):
    shapes = [a.shape for a in arrs]
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc

    return out


def range_offsets(items, item_len=len):
    offsets, off = {}, 0

    for key, item in items:
        offsets[key] = off
        off += item_len(item)

    return offsets, off


def fromarrays(**fields):
    arrs = [np.asarray(v) for v in fields.values()]
    dtype = np.dtype([(n, a.dtype, a.shape[1:]) for n, a in zip(fields, arrs)])
    out = np.empty(len(arrs[0]), dtype=dtype)
    for n, a in zip(fields, arrs):
        out[n] = a

    return out


def clean(origfn=None, tol=1e-10, ckwarg='clean'):
    def cleanfn(fn):
        @ft.wraps(fn)
        def newfn(*args, **kwargs):
            if not kwargs.pop(ckwarg, True):
                return fn(*args, **kwargs)

            arr = fn(*args, **kwargs).copy()

            # Flush small elements to zero
            arr[np.abs(arr) < tol] = 0

            # Coalesce similar elements
            if arr.size > 1:
                amfl = np.abs(arr.flat)
                amix = np.argsort(amfl)
                atol = 0.1*tol

                i, ix = 0, amix[0]
                for j, jx in enumerate(amix[1:], start=1):
                    if abs(amfl[jx] - amfl[ix]) > atol + tol*amfl[ix]:
                        if j - i > 1:
                            amfl[amix[i:j]] = np.median(amfl[amix[i:j]])
                        i, ix = j, jx

                if i != j:
                    amfl[amix[i:]] = np.median(amfl[amix[i:]])

                # Fix up the signs and assign
                arr.flat = np.copysign(amfl, arr.flat)

            return arr
        return newfn

    return cleanfn(origfn) if origfn else cleanfn


def morton_encode(ipts, imax, dtype=np.uint64):
    # Allocate the codes
    codes = np.zeros(len(ipts), dtype=dtype)

    # Determine how many bits to use for each input dimension
    ndims = ipts.shape[1]
    obits = 8*codes.dtype.itemsize
    ibits = obits // ndims
    ishift = np.array([max(int(p).bit_length() - ibits, 0) for p in imax],
                      dtype=dtype)

    # Compute the masks and shifts
    ops = [[(1 << j, (ndims - 1)*j + i) for j in range(ibits)]
           for i in range(ndims)]

    # Cache-block the arrays
    n = max(1, len(codes) // 16384)
    bipts = np.array_split(ipts, n)
    bcodes = np.array_split(codes, n)

    # Loop over each block
    for ipt, code in zip(bipts, bcodes):
        # Loop over each dimension
        for p, pops in zip((ipt >> ishift).T, ops):
            # Extract and interleave the bits
            for mask, shift in pops:
                code |= (p & mask) << shift

    return codes


def search_unsorted(a, v):
    idx = np.argsort(a)
    return idx[np.searchsorted(a, v, sorter=idx)]


def fuzzysort(arr, idx, dim=0, tol=1e-6):
    # Extract our dimension and argsort
    arrd = arr[dim]
    srtdidx = sorted(idx, key=arrd.__getitem__)

    if len(srtdidx) > 1:
        i, ix = 0, srtdidx[0]
        for j, jx in enumerate(srtdidx[1:], start=1):
            if arrd[jx] - arrd[ix] >= tol:
                if j - i > 1:
                    srtdidx[i:j] = fuzzysort(arr, srtdidx[i:j], dim + 1, tol)
                i, ix = j, jx

        if i != j:
            srtdidx[i:] = fuzzysort(arr, srtdidx[i:], dim + 1, tol)

    return srtdidx


def _batched_fuzzysort_rec(coords, perm, dim, s, e, ndims, tol):
    neles = perm.shape[0]
    group = perm[:, s:e]
    vals = np.take_along_axis(coords[:, dim, :], group, axis=1)

    sub = np.argsort(vals, axis=1, kind='stable')
    perm[:, s:e] = np.take_along_axis(group, sub, axis=1)

    if dim + 1 >= ndims:
        return np.zeros(neles, dtype=bool)

    # Find tolerance-based group boundaries from the first element
    sorted_vals = np.take_along_axis(vals, sub, axis=1)
    ref_gaps = np.diff(sorted_vals[0]) >= tol

    # Tag elements that have different group boundaries
    bad = np.any((np.diff(sorted_vals, axis=1) >= tol) != ref_gaps, axis=1)

    # Recursively sub-sort group by the next dimension
    prev = 0
    for b in [*(np.flatnonzero(ref_gaps) + 1), e - s]:
        if b - prev > 1:
            bad |= _batched_fuzzysort_rec(coords, perm, dim + 1,
                                          s + prev, s + b, ndims, tol)
        prev = b

    return bad


def batched_fuzzysort(coords, tol=1e-6):
    neles, ndims, nfp = coords.shape

    if nfp <= 1:
        return np.zeros((neles, 1), dtype=int)

    perm = np.tile(np.arange(nfp), (neles, 1))
    bad = _batched_fuzzysort_rec(coords, perm, 0, 0, nfp, ndims, tol)

    # Handle pathological elements
    for bi in np.flatnonzero(bad):
        perm[bi] = fuzzysort(coords[bi], list(range(nfp)), tol=tol)

    return perm


def iter_struct(arr, n=1000, axis=0):
    for c in np.array_split(arr, -(arr.shape[axis] // -n) or 1, axis=axis):
        yield from c.tolist()


bfloat16 = np.dtype('u2', metadata={'pyfr_bf16': True})


def is_bf16(dtype):
    md = np.dtype(dtype).metadata
    return bool(md and md.get('pyfr_bf16'))


def f32_to_bf16(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    b = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

    # Handle NaN
    b[np.isnan(x)] = 0x7FC0
    return b


def bf16_to_f32(u):
    u32 = np.ascontiguousarray(u, dtype=np.uint16).astype(np.uint32) << 16
    return u32.view(np.float32)


_ctype_map = {
    np.int32: 'int', np.uint32: 'unsigned int',
    np.int64: 'int64_t', np.uint64: 'uint64_t',
    np.float16: 'half',
    np.float32: 'float', np.float64: 'double'
}


def npdtype_to_ctype(dtype):
    dtype = np.dtype(dtype)
    if is_bf16(dtype):
        return 'bf16'
    else:
        return _ctype_map[dtype.type]


_ctypestype_map = {
    np.int32: ct.c_int32, np.uint32: ct.c_uint32,
    np.int64: ct.c_int64, np.uint64: ct.c_uint64,
    np.float32: ct.c_float, np.float64: ct.c_double
}


def npdtype_to_ctypestype(dtype):
    # Special-case None which otherwise expands to np.float
    if dtype is None:
        return None

    dt = np.dtype(dtype)
    ctype = _ctypestype_map[dt.base.type]
    if dt.shape:
        ctype = ctype*prod(dt.shape)

    return ctype


class BLASThreadCtrl:
    _instance = None

    def __init__(self):
        self._set_single = lambda: None
        self._restore = lambda token: None

        try:
            blas = np.show_config(mode='dicts')['Build Dependencies']['blas']
            name = blas['name'].lower()
            if 'openblas' in name:
                self._init_openblas()
            elif name == 'accelerate':
                self._init_accelerate()
        except (AttributeError, KeyError, OSError, StopIteration):
            pass

    def _init_openblas(self):
        nploc = Path(np.__file__).parent
        if sys.platform == 'darwin':
            libdir, ext = nploc / '.dylibs', 'dylib*'
        elif sys.platform == 'win32':
            libdir, ext = nploc.parent / 'numpy.libs', 'dll'
        else:
            libdir, ext = nploc.parent / 'numpy.libs', 'so*'

        path = next(libdir.glob(f'libscipy_openblas*.{ext}'))
        set_fn = ct.CDLL(str(path)).openblas_set_num_threads_local
        set_fn.argtypes = [ct.c_int]

        self._set_single = lambda: set_fn(1)
        self._restore = set_fn

    def _init_accelerate(self):
        lib = ct.CDLL('/System/Library/Frameworks/Accelerate.framework/'
                      'Accelerate')
        set_fn, get_fn = lib.BLASSetThreading, lib.BLASGetThreading
        set_fn.argtypes = [ct.c_uint]
        get_fn.argtypes, get_fn.restype = [], ct.c_uint

        def set_single():
            prev = get_fn()
            set_fn(1)
            return prev

        self._set_single = set_single
        self._restore = set_fn

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    @classmethod
    @contextlib.contextmanager
    def serial(cls):
        self = cls.instance()
        nth = self._set_single()
        try:
            yield
        finally:
            self._restore(nth)


class GPOptimiser:
    _ls_pad = 0.1
    _base_noise = 0.01

    def __init__(self, wsize, x_bounds):
        self._X, self._y = np.empty((2, wsize))
        self._nv = np.full(wsize, self._base_noise)
        self._seq = np.empty(wsize, dtype=int)
        self.n = 0
        self._idx = self._step = 0
        self.x_lo, self.x_hi = x_bounds

    def reset(self, x_bounds=None):
        self.n = 0
        self._idx = self._step = 0
        if x_bounds is not None:
            self.x_lo, self.x_hi = x_bounds

    def record(self, x, y, noise_var=None):
        self._X[self._idx] = x
        self._y[self._idx] = y
        nv = self._base_noise if noise_var is None else noise_var
        self._nv[self._idx] = nv
        self._seq[self._idx] = self._step
        self._step += 1
        self._idx = (self._idx + 1) % len(self._X)
        self.n = min(self.n + 1, len(self._X))

    def update(self, x, y, noise_var=None, tol=0.05):
        nearby = np.flatnonzero(np.abs(self._X[:self.n] - x) < tol)

        if len(nearby):
            newest = nearby[self._seq[nearby].argmax()]
            self._y[newest] = y
            if noise_var is not None:
                self._nv[newest] = noise_var
            self._seq[newest] = self._step
            self._step += 1
        else:
            self.record(x, y, noise_var)

    def _norm_cdf(self, z):
        return np.array([0.5*(1.0 + erf(zi*2**-0.5)) for zi in z])

    def _norm_pdf(self, z):
        return np.exp(-0.5*z**2) / (2*np.pi)**0.5

    def _fit(self):
        X, y = self._X[:self.n], self._y[:self.n]
        nv_diag = np.diag(self._nv[:self.n])
        y_mean = y.mean()
        y_c = y - y_mean

        ls = 0.5*(np.ptp(X) + self._ls_pad)

        K = np.exp(-0.5*((X[:, None] - X[None, :])/ls)**2)
        K += nv_diag
        alpha = np.linalg.solve(K, y_c)

        x_test = np.linspace(self.x_lo, self.x_hi, 50)
        k = np.exp(-0.5*((x_test[:, None] - X[None, :])/ls)**2)
        mu = k @ alpha + y_mean

        v = np.linalg.solve(K, k.T)
        var = np.maximum(1.0 - np.sum(k*v.T, axis=1), 1e-10)

        return x_test, mu, var

    def optimum(self, minimise=True, explore=False):
        if self.n < 2:
            return None

        x_test, mu, var = self._fit()

        # Expected Improvement acquisition
        if explore:
            ys = self._y[:self.n]
            f_best = np.min(ys) if minimise else np.max(ys)
            sigma = var**0.5
            imp = (f_best - mu) if minimise else (mu - f_best)
            z = imp / sigma
            ei = imp*self._norm_cdf(z) + sigma*self._norm_pdf(z)
            return x_test[np.argmax(ei)]
        # Pure exploitation; return the GP mean optimum
        else:
            idx = np.argmin(mu) if minimise else np.argmax(mu)
            return x_test[idx]


class LogGPOptimiser(GPOptimiser):
    def __init__(self, wsize, x_bounds):
        super().__init__(wsize, map(np.log, x_bounds))

    def record(self, x, y, noise_var=None):
        super().record(np.log(x), y, noise_var)

    def update(self, x, y, noise_var=None, tol=0.05):
        super().update(np.log(x), y, noise_var, tol)

    def optimum(self, minimise=True, explore=False):
        opt = super().optimum(minimise, explore)
        return np.exp(opt) if opt is not None else None

    def reset(self, x_bounds=None):
        if x_bounds is not None:
            x_bounds = map(np.log, x_bounds)
        super().reset(x_bounds)
