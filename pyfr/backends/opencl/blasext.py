from pyfr.backends.base.blasext import BaseBlasExtKernels
from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider


class OpenCLBlasExtKernels(BaseBlasExtKernels, OpenCLKernelProvider):
    def copy(self, dst, src):
        cl = self.backend.cl

        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(OpenCLKernel):
            def run(self, queue, wait_for=None, ret_evt=False):
                return cl.memcpy(queue, dst, src, dst.nbytes, blocking=False,
                                 wait_for=wait_for, ret_evt=ret_evt)

        return CopyKernel(mats=[dst, src])

    def zero(self, m):
        cl = self.backend.cl

        class ZeroKernel(OpenCLKernel):
            def run(self, queue, wait_for=None, ret_evt=False):
                return cl.zero(m, m.nbytes, queue, wait_for=wait_for,
                               ret_evt=ret_evt)

        return ZeroKernel(mats=[m])
