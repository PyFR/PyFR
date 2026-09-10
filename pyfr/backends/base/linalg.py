import numpy as np

from pyfr.backends.base.provider import NotSuitableError
from pyfr.nputil import is_bf16, npdtype_to_ctype


class BaseLinalgKernels:
    @staticmethod
    def batched_inv_ok_dtypes(mdtype, odtype):
        mdtype, odtype = np.dtype(mdtype), np.dtype(odtype)
        o16 = odtype == np.float16 or is_bf16(odtype)
        return mdtype == odtype or (mdtype == np.float32 and o16)

    @staticmethod
    def check_inv_eidxs(curr, new, nemax):
        # Check the rebound index window fits within the staging capacity
        if new.dtype != curr.dtype or new.ncol > nemax:
            raise ValueError('Rebound eidxs must match type and fit capacity')

    @staticmethod
    def apply_tiled_tplargs(minv, *, nupts, nvars, in_scale, out_scale):
        return {
            'nupts': nupts, 'nvars': nvars, 'block_size': minv.block_size,
            'trows': minv.trows, 'tcols': minv.tcols,
            'ntiles_c': minv.ntiles_c,
            'in_scale': in_scale, 'out_scale': out_scale,
            'pcdtype': npdtype_to_ctype(minv.dtype)
        }


class BaseGPULinalgKernels(BaseLinalgKernels):
    # Threads per block for the tiled matvec
    _mv_nthreads = 64

    def batched_inv_tiled(self, m, out, *, eidxs):
        if (not self.batched_inv_ok_dtypes(m.dtype, out.dtype) or
            m.ioshape[0] != out.block_size):
            raise NotSuitableError('Incompatible tiled inverse output')

        return self._batched_inv(m, out, eidxs=eidxs)

    def batched_tiled_matvec(self, x, minv, y, *, nupts, nvars, in_scale=(),
                             out_scale=()):
        tplargs = self.apply_tiled_tplargs(minv, nupts=nupts, nvars=nvars,
                                           in_scale=in_scale,
                                           out_scale=out_scale)
        tplargs['mvthreads'] = self._mv_nthreads
        return self._batched_tiled_matvec(x, minv, y, tplargs)

    def _compute_dtype(self, mdtype):
        return np.dtype(np.float32 if is_bf16(mdtype) else mdtype)

    def _base_tplargs(self, out, mdtype):
        return {
            'n': out.block_size, 'trows': out.trows, 'tcols': out.tcols,
            'padr': out.padr, 'padc': out.padc, 'sgsize': self._sgsize,
            'ftype': npdtype_to_ctype(self._compute_dtype(mdtype)),
            'stype': npdtype_to_ctype(mdtype),
            'otype': npdtype_to_ctype(out.dtype),
        }

    def _inv_sc(self, n, mdtype):
        return min(n, 64 if np.dtype(mdtype).itemsize == 8 else 128)

    def _inv_params(self, n):
        if n < 128:
            return {'fblk': 8, 'sblk': 8, 'nthreads': 64}
        else:
            return {'fblk': 16, 'sblk': 16, 'nthreads': 128}

    def _inv_tiles(self, mdtype):
        if np.dtype(mdtype).itemsize == 4:
            return 8, 4, 2, 2
        else:
            return 2, 2, 2, 2

    def _inv_tplargs(self, out, mdtype):
        n = out.block_size
        cdtype = self._compute_dtype(mdtype)
        stm, stn, ftm, ftn = self._inv_tiles(cdtype)
        extra = {'sc': self._inv_sc(n, cdtype), 'stm': stm, 'stn': stn,
                 'ftm': ftm, 'ftn': ftn}
        return self._base_tplargs(out, mdtype) | self._inv_params(n) | extra

    def _inv_argtypes(self, mdtype):
        ixdtype = self.backend.ixdtype
        cdtype = self._compute_dtype(mdtype)

        return [np.uintp]*5 + [ixdtype, ixdtype, np.dtype(cdtype).type]

    def _compile_inv(self, out, mdtype):
        tplargs = self._inv_tplargs(out, mdtype)
        tpl = self.backend.lookup.get_template(
            'pyfr.backends.base.kernels.lu_inv'
        )
        fn = self._build_kernel('lu_inv', tpl.render(**tplargs),
                                self._inv_argtypes(mdtype))
        return fn, tplargs['nthreads']
