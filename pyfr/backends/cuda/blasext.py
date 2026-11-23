from pyfr.backends.base.blasext import BaseBlasExtKernels
from pyfr.backends.cuda.provider import CUDAKernel, CUDAKernelProvider


class CUDABlasExtKernels(BaseBlasExtKernels, CUDAKernelProvider):
    def copy(self, dst, src):
        cuda = self.backend.cuda

        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(CUDAKernel):
            def add_to_graph(self, graph, deps):
                return graph.graph.add_memcpy(dst, src, dst.nbytes, deps)

            def run(self, stream):
                cuda.memcpy(dst, src, dst.nbytes, stream)

        return CopyKernel(mats=[dst, src])

    def zero(self, m):
        cuda = self.backend.cuda

        class ZeroKernel(CUDAKernel):
            def add_to_graph(self, graph, deps):
                return graph.graph.add_memset(m, 0, m.nbytes, deps)

            def run(self, stream):
                cuda.memset(m, 0, m.nbytes, stream)

        return ZeroKernel(mats=[m])
