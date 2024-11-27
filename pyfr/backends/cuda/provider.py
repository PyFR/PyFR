from weakref import WeakKeyDictionary

from pyfr.backends.base import (BaseKernelProvider, BaseOrderedMetaKernel,
                                BasePointwiseKernelProvider,
                                BaseUnorderedMetaKernel, Kernel)
from pyfr.backends.cuda.compiler import CUDACompilerModule
from pyfr.backends.cuda.generator import CUDAKernelGenerator
from pyfr.cache import memoize


def get_grid_for_block(block, nrow, ncol=1):
    return (-(-nrow // block[0]), -(-ncol // block[1]), 1)


class CUDAKernel(Kernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if hasattr(self, 'bind') and hasattr(self, 'add_to_graph'):
            self.gnodes = WeakKeyDictionary()


class CUDAOrderedMetaKernel(BaseOrderedMetaKernel):
    def add_to_graph(self, graph, dnodes):
        for k in self.kernels:
            dnodes = [k.add_to_graph(graph, dnodes)]

        return dnodes[0]


class CUDAUnorderedMetaKernel(BaseUnorderedMetaKernel):
    def add_to_graph(self, graph, dnodes):
        nodes = [k.add_to_graph(graph, dnodes) for k in self.kernels]

        return graph.graph.add_empty(nodes)


class CUDAKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes, argn=[]):
        mod = CUDACompilerModule(self.backend, src)
        return mod.get_function(name, argtypes)

    def _benchmark(self, kfunc, nbench=4, nwarmup=1):
        stream = self.backend.cuda.create_stream()
        start_evt = self.backend.cuda.create_event(timing=True)
        stop_evt = self.backend.cuda.create_event(timing=True)

        for i in range(nbench + nwarmup):
            if i == nwarmup:
                start_evt.record(stream)

            kfunc(stream)

        stop_evt.record(stream)
        stream.synchronize()

        return stop_evt.elapsed_time(start_evt) / nbench


class CUDAPointwiseKernelProvider(CUDAKernelProvider,
                                  BasePointwiseKernelProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._block1d = (64, 1, 1)
        self._block2d = (32, 8, 1)

        # Pass these block sizes to the generator
        class KernelGenerator(CUDAKernelGenerator):
            block1d = self._block1d
            block2d = self._block2d

        self.kernel_generator_cls = KernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst, argm, argv):
        rtargs = []
        block = self._block1d if len(dims) == 1 else self._block2d
        grid = get_grid_for_block(block, dims[-1])

        # Set shared memory carveout locally for kernel
        fun.set_shared_size(carveout=25 if fun.shared_mem else 0)

        params = fun.make_params(grid, block)

        # Process the arguments
        for i, k in enumerate(arglst):
            if isinstance(k, str):
                rtargs.append((i, k))
            else:
                params.set_arg(i, k)

        class PointwiseKernel(CUDAKernel):
            if rtargs:
                def bind(self, **kwargs):
                    for i, k in rtargs:
                        params.set_arg(i, kwargs[k])

                    # Notify any graphs we're in about our new parameters
                    for graph, gnode in self.gnodes.items():
                        graph.stale_kparams[gnode] = params

            def add_to_graph(self, graph, deps):
                gnode = graph.graph.add_kernel(params, deps)

                # If our parameters can change then we need to keep a
                # (weak) reference to the graph so we can notify it
                if rtargs:
                    self.gnodes[graph] = gnode

                return gnode

            def run(self, stream):
                fun.exec_async(stream, params)

        return PointwiseKernel(argm, argv)
