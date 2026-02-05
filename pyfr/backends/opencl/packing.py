import numpy as np

from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider


class OpenCLPackingKernels(OpenCLKernelProvider):
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
        kern.set_dims((v.n,))
        kern.set_args(v.n, v.nvrow, v.nvcol, v.basedata, v.mapping,
                      v.rstrides or 0, xm)

        return kern

    def pack(self, xmv):
        cl = self.backend.cl
        xm, v = self._extract_args(xmv)

        # If we have a view then obtain a view-packing kernel
        if v:
            kern = self._packing_kern('pack', v, xm)

            class PackKernel(OpenCLKernel):
                def run(self, queue, wait_for=None, ret_evt=False):
                    pevt = kern.exec_async(queue, wait_for, True)
                    return cl.memcpy(queue, xm.hdata, xm.data, xm.nbytes,
                                     False, None, ret_evt)
        else:
            class PackKernel(OpenCLKernel):
                def run(self, queue, wait_for=None, ret_evt=False):
                    return cl.memcpy(queue, xm.hdata, xm.data, xm.nbytes,
                                     False, wait_for, ret_evt)

        return PackKernel(mats=[xmv])

    def unpack(self, xmv):
        cl = self.backend.cl
        xm, v = self._extract_args(xmv)

        # If we have an exchange view then obtain a view-unpacking kernel
        if v:
            kern = self._packing_kern('unpack', v, xm)

            class UnpackKernel(OpenCLKernel):
                def run(self, queue, wait_for=None, ret_evt=False):
                    mevt = cl.memcpy(queue, xm.data, xm.hdata, xm.nbytes,
                                     False, wait_for, True)
                    return kern.exec_async(queue, [mevt], ret_evt)
        else:
            class UnpackKernel(OpenCLKernel):
                def run(self, queue, wait_for=None, ret_evt=False):
                    return cl.memcpy(queue, xm.data, xm.hdata, xm.nbytes,
                                     False, wait_for, ret_evt)

        return UnpackKernel(mats=[xmv])
