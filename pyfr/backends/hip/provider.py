from weakref import WeakKeyDictionary

from pyfr.backends.base import (BaseKernelProvider, BaseOrderedMetaKernel,
                                BasePointwiseKernelProvider,
                                BaseUnorderedMetaKernel, Kernel)
from pyfr.backends.hip.compiler import HIPCompilerModule
from pyfr.backends.hip.generator import HIPKernelGenerator
from pyfr.cache import memoize


def get_grid_for_block(block, nrow, ncol=1):
    return (-(-nrow // block[0]), -(-ncol // block[1]), 1)


class HIPKernel(Kernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if hasattr(self, 'bind') and hasattr(self, 'add_to_graph'):
            self.gnodes = WeakKeyDictionary()


class HIPOrderedMetaKernel(BaseOrderedMetaKernel):
    def add_to_graph(self, graph, dnodes):
        pass


class HIPUnorderedMetaKernel(BaseUnorderedMetaKernel):
    def add_to_graph(self, graph, dnodes):
        pass


class HIPKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes, argn=[]):
        mod = HIPCompilerModule(self.backend, src)
        return mod.get_function(name, argtypes)

    def _benchmark(self, kfunc, nbench=4, nwarmup=1):
        stream = self.backend.hip.create_stream()
        start_evt = self.backend.hip.create_event()
        stop_evt = self.backend.hip.create_event()

        for i in range(nbench + nwarmup):
            if i == nwarmup:
                start_evt.record(stream)

            kfunc(stream)

        stop_evt.record(stream)
        stream.synchronize()

        return stop_evt.elapsed_time(start_evt) / nbench


class HIPPointwiseKernelProvider(HIPKernelProvider,
                                 BasePointwiseKernelProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._block1d = (64, 1, 1)
        self._block2d = (64, 4, 1)

        # Pass these block sizes to the generator
        class KernelGenerator(HIPKernelGenerator):
            block1d = self._block1d
            block2d = self._block2d

        self.kernel_generator_cls = KernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst, argm, argv):
        rtargs = []
        block = self._block1d if len(dims) == 1 else self._block2d
        grid = get_grid_for_block(block, dims[-1])

        params = fun.make_params(grid, block)

        # Process the arguments
        for i, k in enumerate(arglst):
            if isinstance(k, str):
                rtargs.append((i, k))
            else:
                params.set_arg(i, k)

        class PointwiseKernel(HIPKernel):
            if rtargs:
                def bind(self, **kwargs):
                    for i, k in rtargs:
                        params.set_arg(i, kwargs[k])

            def add_to_graph(self, graph, deps):
                pass

            def run(self, stream):
                fun.exec_async(stream, params)

        return PointwiseKernel(argm, argv)
