from pyfr.backends.base.blasext import BaseBlasExtKernels
from pyfr.backends.metal.provider import MetalKernel, MetalKernelProvider
from pyfr.backends.metal.util import call_


class MetalBlasExtKernels(BaseBlasExtKernels, MetalKernelProvider):
    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(MetalKernel):
            def run(self, cbuf):
                blit = cbuf.blitCommandEncoder()
                call_(blit, 'copy', fromBuffer=src.basedata,
                      sourceOffset=src.offset, toBuffer=dst.basedata,
                      destinationOffset=dst.offset, size=dst.nbytes)
                blit.endEncoding()

        return CopyKernel(mats=[dst, src])

    def zero(self, m):
        from Metal import NSMakeRange

        class ZeroKernel(MetalKernel):
            def run(self, cbuf):
                blit = cbuf.blitCommandEncoder()
                blit.fillBuffer_range_value_(m.basedata,
                                             NSMakeRange(m.offset, m.nbytes),
                                             0)
                blit.endEncoding()

        return ZeroKernel(mats=[m])
