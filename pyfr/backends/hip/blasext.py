from pyfr.backends.base.blasext import BaseBlasExtKernels
from pyfr.backends.hip.provider import HIPKernel, HIPKernelProvider


class HIPBlasExtKernels(BaseBlasExtKernels, HIPKernelProvider):
    def copy(self, dst, src):
        hip = self.backend.hip

        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(HIPKernel):
            def add_to_graph(self, graph, deps):
                return graph.graph.add_memcpy(dst, src, dst.nbytes, deps)

            def run(self, stream):
                hip.memcpy(dst, src, dst.nbytes, stream)

        return CopyKernel(mats=[dst, src])

    def zero(self, m):
        hip = self.backend.hip

        class ZeroKernel(HIPKernel):
            def add_to_graph(self, graph, deps):
                return graph.graph.add_memset(m, 0, m.nbytes, deps)

            def run(self, stream):
                hip.memset(m, 0, m.nbytes, stream)

        return ZeroKernel(mats=[m])
