from functools import cached_property
import os
import warnings

import numpy as np

from pyfr.nputil import is_bf16, npdtype_to_ctype


class LUInverse:
    # Most matrices inverted in one launch, each with its own workspace
    max_nwork = 512

    def __init__(self, backend, m, out, idxs, extent):
        self.backend = backend
        self.m, self.out, self.idxs = m, out, idxs

        # Factorise matrices staged as bf16 in single precision
        self.cdtype = np.dtype(np.float32 if is_bf16(m.dtype) else m.dtype)

        self.tplargs = tplargs = self._tplargs(m, out)
        wsz, nwork, npad = tplargs['wsz'], tplargs['nwork'], tplargs['npad']
        backend.pointwise.register('pyfr.integrators.implicit.kernels.lu_inv')

        # Give each item inverted at once a part of the workspace
        wk = np.arange(nwork)[None, :]
        self.ws = backend.matrix((1, nwork*wsz), dtype=self.cdtype,
                                 extent=extent, tags={'align'})
        self.wk = backend.const_matrix(wk, dtype=backend.ixdtype,
                                       tags={'align'})

        # Hold the row order of each item inverted at once beside its workspace
        self.perm = backend.matrix((1, nwork*npad), dtype=np.int16,
                                   extent=extent, tags={'align'})

    @cached_property
    def parts(self):
        # Split the staged matrices of each index set into parts
        return {k: self._split(ix) for k, ix in self.idxs.items()}

    @cached_property
    def kerns(self):
        # Build one kernel for each number of items inverted at once
        sized = {v.n: (m, v) for ps in self.parts.values() for m, v in ps}
        kerns = {}
        for s, (src, inv) in sized.items():
            kerns[s] = self.backend.kernel(
                'lu_inv', self.tplargs, dims=[s], src=src, inv=inv, scale=1.0,
                ws=self.ws, perm=self.perm, wk=self.wk
            )

        return kerns

    def _tplargs(self, m, out):
        backend, isz = self.backend, self.cdtype.itemsize
        n, sg = out.nrow, backend.subgrpsz

        # Choose the pivot block size, lanes per item and register tile
        if sg == 1:
            blk, nthreads, tm, tn = 16, 1, 2, 16
        else:
            blk = 16 if n >= 128 else 8
            nthreads = 2*sg if blk == 8 and sg < 64 else 4*sg
            tm, tn = 4, 4

        # Split the columns evenly into the fewest chunks of at most 512 bytes
        nblk = -(-n // blk)
        nchunk = -(-nblk // (512 // (isz*blk)))
        npad, sc = nblk*blk, -(-nblk // nchunk)*blk

        # Take the rows per Schur pass which give every lane a register tile
        frb = (nthreads*tm*tn // sc) // blk*blk

        # Limit them to the rows of the padded matrix
        frb = max(blk, min(frb, npad))

        # Hold the padded factor and a chunk of right-hand sides in a workspace
        wsz = npad*(npad + sc)
        nwork = min(self.max_nwork, m.ioshape[2])

        return {
            'n': n, 'npad': npad,
            'ftype': npdtype_to_ctype(self.cdtype),
            'stype': npdtype_to_ctype(m.dtype),
            'otype': npdtype_to_ctype(out.dtype),
            'sc': sc, 'blk': blk, 'frb': frb, 'nthreads': nthreads,
            'tm': tm, 'tn': tn, 'wsz': wsz, 'nwork': nwork
        }

    def _split(self, ix):
        backend, m, out = self.backend, self.m, self.out
        n, nwork = out.nrow, self.tplargs['nwork']

        # View the inverses of the staged matrices
        mid, row = np.full(len(ix), out.mid), np.zeros(len(ix), dtype=int)
        view = backend.view(backend.view_mapping(mid, row, ix))

        # Pair the staged matrices and inverses a workspace at a time
        spans = [(p, min(p + nwork, view.n)) for p in range(0, view.n, nwork)]

        return [(m.slice(ca=p*n, cb=q*n), view.slice(p, q)) for p, q in spans]

    def invert(self, key, scale):
        # Invert the staged matrices of each part
        for src, inv in self.parts[key]:
            kern = self.kerns[inv.n]
            kern.bind(src=src, inv=inv, scale=scale)
            self.backend.run_kernels([kern])


def get_lu_inverse(backend, m, out, idxs, extent):
    # On HIP the inverse may come from hand written kernels instead
    if backend.name == 'hip':
        cfgimpl = backend.cfg.get('backend-hip', 'lu-inverse', 'mako')
        impl = os.environ.get('PYFR_HIP_LU_INVERSE') or cfgimpl

        # Note an override of the config file, for reproducibility
        if impl != cfgimpl:
            warnings.warn(f'PYFR_HIP_LU_INVERSE={impl} overrides the '
                          f'configured lu-inverse = {cfgimpl}')

        if impl == 'hip':
            from pyfr.backends.hip.luinv import HIPLUInverse

            if not HIPLUInverse.supported(backend):
                arch = backend.props['gcn_arch_name']
                raise ValueError(f'HIP LU inverse needs a GPU with MFMA '
                                 f'(gfx90a, gfx94x or gfx95x), found {arch}')

            return HIPLUInverse(backend, m, out, idxs, extent)
        elif impl != 'mako':
            raise ValueError(f'Invalid LU inverse {impl!r}')

    return LUInverse(backend, m, out, idxs, extent)
