import numpy as np

from pyfr.backends.metal.provider import MetalKernel, MetalKernelProvider


class MetalPackingKernels(MetalKernelProvider):
    def _extract_args(self, xmv):
        if isinstance(xmv, self.backend.xchg_view_cls):
            return xmv.xchgmat, xmv.view
        else:
            return xmv, None

    def _packing_kern(self, type, v, xm):
        # Render the kernel template
        src = self.backend.lookup.get_template('packing').render()

        # Build
        kern = self._build_kernel(f'{type}_view', src,
                                  [self.backend.ixdtype]*3 + [np.uintp]*4)
        grid, tgrp = (v.n, 1, 1), (128, 1, 1)

        # Arguments
        n, nvrow, nvcol = v.n, v.nvrow, v.nvcol
        vb, vm = (v.basedata, 0), v.mapping.data
        vr, xd = v.rstrides.data if v.rstrides else (None, 0), xm.data

        return kern, (grid, tgrp, n, nvrow, nvcol, vb, vm, vr, xd)

    def pack(self, xmv):
        xm, v = self._extract_args(xmv)

        if v:
            # Obtain the view-packing kernel
            kern, kargs = self._packing_kern('pack', v, xm)

        class PackKernel(MetalKernel):
            def run(self, cbuf):
                # If necessary call the packing kernel
                if v:
                    kern(cbuf, *kargs)

                # Ensure the host buffer is in sync with the device
                blit = cbuf.blitCommandEncoder()
                blit.synchronizeResource_((v.basedata, 0))
                blit.endEncoding()

        return PackKernel(mats=[xmv])

    def unpack(self, xmv):
        xm, v = self._extract_args(xmv)

        # If we have an exchange view then obtain a view-unpacking kernel
        if v:
            kern, kargs = self._packing_kern('unpack', v, xm)

            class UnpackKernel(MetalKernel):
                def run(self, cbuf):
                    xm.basedata.didModifyRange_((xm.offset, xm.nbytes))
                    kern(cbuf, *kargs)
        else:
            class UnpackKernel(MetalKernel):
                def run(self, cbuf):
                    xm.basedata.didModifyRange_((xm.offset, xm.nbytes))

        return UnpackKernel(mats=[xmv])
