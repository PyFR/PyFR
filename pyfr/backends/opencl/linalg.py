import numpy as np

from pyfr.backends.base.linalg import BaseGPULinalgKernels
from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider


class OpenCLLinalgKernels(BaseGPULinalgKernels, OpenCLKernelProvider):
    _sgsize = 1
    _mv_nthreads = 128

    def _batched_inv(self, m, out, *, eidxs):
        n, nemax = out.block_size, m.ioshape[2]
        cdtype = self._compute_dtype(m.dtype)
        itemsz = np.dtype(cdtype).itemsize

        inv_kern, nthreads = self._compile_inv(out, m.dtype)

        cl = self.backend.cl
        sc = self._inv_sc(n, cdtype)

        chunk = min(nemax, inv_kern.resident_workgroups(nthreads))
        anbytes, cnbytes = chunk*itemsz*n*n, chunk*itemsz*n*sc

        kern = inv_kern.clone()
        kern.set_args(out.data, m.data, eidxs.data, m.leaddim, start=2)
        kern.set_arg(7, 1.0)

        check_inv_eidxs = self.check_inv_eidxs
        ecurr = eidxs

        class BatchedInvKernel(OpenCLKernel):
            def bind(self, *, scale=None, eidxs=None):
                nonlocal ecurr

                if scale is not None:
                    kern.set_arg(7, scale)
                if eidxs is not None:
                    check_inv_eidxs(ecurr, eidxs, nemax)
                    kern.set_arg(4, eidxs.data)
                    ecurr = eidxs

            def run(self, queue, wait_for=None, ret_evt=False):
                neles = ecurr.ncol
                ascratch = cl.mem_alloc(anbytes)
                cscratch = cl.mem_alloc(cnbytes)
                kern.set_args(ascratch, cscratch)

                evt = None
                for eoff in range(0, neles, chunk):
                    cnt = min(chunk, neles - eoff)
                    kern.set_dims((cnt*nthreads,), (nthreads,))
                    kern.set_arg(6, eoff)
                    evt = kern.exec_async(queue, [evt] if evt else wait_for,
                                          ret_evt or eoff + chunk < neles)

                return evt if ret_evt else None

        return BatchedInvKernel(mats=[m, out, eidxs])

    def _batched_tiled_matvec(self, x, minv, y, tplargs):
        ixdtype = self.backend.ixdtype

        tpl = self.backend.lookup.get_template('batched_tiled_matvec')

        kern = self._build_kernel(
            'batched_tiled_matvec', tpl.render(**tplargs),
            [np.uintp, np.uintp, ixdtype, np.uintp, ixdtype]
        )

        nt = self._mv_nthreads
        kern.set_dims((nt*minv.nmats,), (nt,))
        kern.set_args(minv.data, x.data, x.leaddim, y.data, y.leaddim)

        class ApplyTiledMatrixKernel(OpenCLKernel):
            def run(self, queue, wait_for=None, ret_evt=False):
                return kern.exec_async(queue, wait_for, ret_evt)

        return ApplyTiledMatrixKernel(mats=[x, minv, y])
