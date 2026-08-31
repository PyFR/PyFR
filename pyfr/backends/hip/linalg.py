import numpy as np

from pyfr.backends.base.linalg import BaseGPULinalgKernels
from pyfr.backends.hip.provider import HIPKernel, HIPKernelProvider


class HIPLinalgKernels(BaseGPULinalgKernels, HIPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        self._sgsize = backend.props['warp_size']

    def _inv_tiles(self, mdtype):
        if np.dtype(mdtype).itemsize == 4:
            return 4, 4, 2, 2
        else:
            return 2, 2, 2, 2

    def _inv_tplargs(self, out, mdtype):
        tplargs = super()._inv_tplargs(out, mdtype)
        cdtype = self._compute_dtype(mdtype)
        tplargs['nthreads'] = 256 if cdtype.itemsize == 8 else 128
        return tplargs

    def _batched_inv(self, m, out, *, eidxs):
        n, nemax = out.block_size, m.ioshape[2]
        cdtype = self._compute_dtype(m.dtype)
        itemsz = np.dtype(cdtype).itemsize

        kern, nthreads = self._compile_inv(out, m.dtype)

        hip = self.backend.hip
        ncu = self.backend.props['multiprocessor_count']
        chunk = min(nemax, kern.resident_blocks(nthreads, ncu))
        sc = self._inv_sc(n, cdtype)
        anbytes, cnbytes = chunk*itemsz*n*n, chunk*itemsz*n*sc

        params = kern.make_params((chunk, 1, 1), (nthreads, 1, 1))
        params.set_args(out, m, eidxs, m.leaddim, start=2)
        params.set_arg(7, 1.0)

        check_inv_eidxs = self.check_inv_eidxs
        ecurr = eidxs

        class BatchedInvKernel(HIPKernel):
            def bind(self, *, scale=None, eidxs=None):
                nonlocal ecurr

                if scale is not None:
                    params.set_arg(7, scale)
                if eidxs is not None:
                    check_inv_eidxs(ecurr, eidxs, nemax)
                    params.set_arg(4, eidxs)
                    ecurr = eidxs

            def run(self, stream):
                buf = hip.mem_alloc(anbytes + cnbytes, stream)
                params.set_args(int(buf), int(buf) + anbytes)

                for eoff in range(0, ecurr.ncol, chunk):
                    params.grid[0] = min(chunk, ecurr.ncol - eoff)
                    params.set_arg(6, eoff)
                    kern.exec_async(stream, params)

                buf.free_async(stream)

        return BatchedInvKernel(mats=[m, out, eidxs])

    def _batched_tiled_matvec(self, x, minv, y, tplargs):
        ixdtype = self.backend.ixdtype

        tpl = self.backend.lookup.get_template('batched_tiled_matvec')

        kern = self._build_kernel(
            'batched_tiled_matvec', tpl.render(**tplargs),
            [np.uintp, np.uintp, ixdtype, np.uintp, ixdtype]
        )

        grid, block = (minv.nmats, 1, 1), (self._mv_nthreads, 1, 1)
        params = kern.make_params(grid, block)
        params.set_args(minv.data, x.data, x.leaddim, y.data,
                        y.leaddim)

        class ApplyTiledMatrixKernel(HIPKernel):
            def run(self, stream):
                kern.exec_async(stream, params)

        return ApplyTiledMatrixKernel(mats=[x, minv, y])
