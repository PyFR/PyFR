import numpy as np

from pyfr.backends.metal.provider import MetalKernel, MetalKernelProvider


class MetalPackingKernels(MetalKernelProvider):
    def pack(self, mv):
        ixdtype = self.backend.ixdtype

        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        src = self.backend.lookup.get_template('pack').render()

        # Build
        kern = self._build_kernel('pack_view', src, [ixdtype]*3 + [np.uintp]*4)
        grid, tgrp = (v.n, 1, 1), (128, 1, 1)

        # Arguments
        n, nvrow, nvcol = v.n, v.nvrow, v.nvcol
        vb, vm = (v.basedata, 0), v.mapping.data
        vr, md = v.rstrides.data if v.rstrides else (None, 0), m.data

        class PackXchgViewKernel(MetalKernel):
            def run(self, cbuf):
                # Call the packing kernel
                kern(cbuf, grid, tgrp, n, nvrow, nvcol, vb, vm, vr, md)

                # Ensure the host buffer is in sync with the device
                blit = cbuf.blitCommandEncoder()
                blit.synchronizeResource_(vb)
                blit.endEncoding()

        return PackXchgViewKernel(mats=[mv])

    def unpack(self, mv):
        class UnpackXchgMatrixKernel(MetalKernel):
            def run(self, cbuf):
                mv.basedata.didModifyRange_((mv.offset, mv.nbytes))

        return UnpackXchgMatrixKernel(mats=[mv])
