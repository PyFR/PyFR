import numpy as np

from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider


class OpenCLPackingKernels(OpenCLKernelProvider):
    def pack(self, mv):
        cl = self.backend.cl

        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        src = self.backend.lookup.get_template('pack').render()

        # Build
        kern = self._build_kernel('pack_view', src, [np.int32]*3 + [np.intp]*4)
        kern.set_dims((v.n,))
        kern.set_args(v.n, v.nvrow, v.nvcol, v.basedata, v.mapping,
                      v.rstrides or 0, m)

        class PackXchgViewKernel(OpenCLKernel):
            def run(self, queue, wait_for=None, ret_evt=False):
                pevt = kern.exec_async(queue, wait_for, True)
                return cl.memcpy(queue, m.hdata, m.data, m.nbytes,
                                 False, [pevt], ret_evt)

        return PackXchgViewKernel(mats=[mv])

    def unpack(self, mv):
        cl = self.backend.cl

        class UnpackXchgMatrixKernel(OpenCLKernel):
            def run(self, queue, wait_for=None, ret_evt=False):
                return cl.memcpy(queue, mv.data, mv.hdata, mv.nbytes,
                                 False, wait_for, ret_evt)

        return UnpackXchgMatrixKernel(mats=[mv])
