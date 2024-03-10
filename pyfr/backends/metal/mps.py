from pyfr.backends.metal.provider import MetalKernel, MetalKernelProvider
from pyfr.backends.metal.util import call_, init_


class MetalMPSKernels(MetalKernelProvider):
    MPSDataTypeFloat32 = 0x10000000 | 32

    def __init__(self, backend):
        super().__init__(backend)

        # Timing data cache
        self._mul_timing = {}

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        from Metal import (MPSMatrix, MPSMatrixDescriptor,
                           MPSMatrixMultiplication)

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        def make_mps_mat(m):
            desc = call_(MPSMatrixDescriptor, 'matrixDescriptorWith',
                         rows=m.nrow, columns=m.ncol,
                         rowBytes=m.leaddim*m.itemsize,
                         dataType=self.MPSDataTypeFloat32)

            return init_(MPSMatrix, buffer=m.basedata, offset=m.offset,
                         descriptor=desc)

        # Create the matrix wrappers
        A = make_mps_mat(a)
        B = make_mps_mat(b)
        C = make_mps_mat(out)

        # Create the multiplication itself
        mm = init_(MPSMatrixMultiplication, device=self.backend.dev,
                   transposeLeft=False, transposeRight=False,
                   resultRows=out.nrow, resultColumns=out.ncol,
                   interiorColumns=a.ncol, alpha=alpha, beta=beta)

        # Cache key
        ckey = (a.dtype, alpha, beta, a.nrow, b.ncol, a.ncol, a.leaddim,
                b.leaddim, out.leaddim)

        # Obtain the performance of the kernel
        try:
            dt = self._mul_timing[ckey]
        except KeyError:
            # Allocate a temporary output buffer
            temp_out = self.backend.matrix(out.ioshape, tags=out.tags)
            temp_mat = make_mps_mat(temp_out)

            def gemm(cbuf):
                call_(mm, 'encodeTo', commandBuffer=cbuf, leftMatrix=A,
                      rightMatrix=B, resultMatrix=temp_mat)

            # Benchmark the kernel and update the cache
            self._mul_timing[ckey] = dt = self._benchmark(gemm)

        class MulKernel(MetalKernel):
            def run(self, cbuf):
                call_(mm, 'encodeTo', commandBuffer=cbuf, leftMatrix=A,
                      rightMatrix=B, resultMatrix=C)

        return MulKernel(mats=[a, b, out], dt=dt)
