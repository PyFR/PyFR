import numpy as np

from pyfr.backends.base.linalg import BaseGPULinalgKernels
from pyfr.backends.metal.provider import MetalKernel, MetalKernelProvider


class MetalLinalgKernels(BaseGPULinalgKernels, MetalKernelProvider):
    _sgsize = 32

    def _batched_inv(self, m, out, *, eidxs):
        n, nemax = out.block_size, m.ioshape[2]
        cdtype = self._compute_dtype(m.dtype)
        itemsz = np.dtype(cdtype).itemsize

        inv_kern, nthreads = self._compile_inv(out, m.dtype)

        mem_alloc = self.backend.mem_alloc
        chunk = min(nemax, 1024)
        sc = self._inv_sc(n, cdtype)
        anbytes, cnbytes = chunk*itemsz*n*n, chunk*itemsz*n*sc

        check_inv_eidxs = self.check_inv_eidxs
        ecurr = eidxs

        class BatchedInvKernel(MetalKernel):
            scale = 1.0

            def bind(self, *, scale=None, eidxs=None):
                nonlocal ecurr

                if scale is not None:
                    self.scale = scale
                if eidxs is not None:
                    check_inv_eidxs(ecurr, eidxs, nemax)
                    ecurr = eidxs

            def run(self, cbuf):
                ascratch, cscratch = mem_alloc(anbytes), mem_alloc(cnbytes)
                for eoff in range(0, ecurr.ncol, chunk):
                    cnt = min(chunk, ecurr.ncol - eoff)
                    inv_kern(cbuf, (cnt*nthreads, 1, 1), (nthreads, 1, 1),
                             (ascratch, 0), (cscratch, 0), out.data, m.data,
                             ecurr.data, m.leaddim, eoff, self.scale)

        return BatchedInvKernel(mats=[m, out, eidxs])

    def _batched_tiled_matvec(self, x, minv, y, tplargs):
        ixdtype = self.backend.ixdtype

        tpl = self.backend.lookup.get_template('batched_tiled_matvec')

        kern = self._build_kernel(
            'batched_tiled_matvec', tpl.render(**tplargs),
            [np.uintp, np.uintp, ixdtype, np.uintp, ixdtype]
        )

        nt = self._mv_nthreads
        grid, tgrp = (minv.nmats*nt, 1, 1), (nt, 1, 1)
        kargs = [minv.data, x.data, x.leaddim, y.data, y.leaddim]

        class ApplyTiledMatrixKernel(MetalKernel):
            def run(self, cbuf):
                kern(cbuf, grid, tgrp, *kargs)

        return ApplyTiledMatrixKernel(mats=[x, minv, y])
