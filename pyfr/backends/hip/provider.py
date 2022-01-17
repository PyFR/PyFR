# -*- coding: utf-8 -*-

from weakref import WeakKeyDictionary

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, Kernel,
                                MetaKernel)
from pyfr.backends.hip.compiler import SourceModule
from pyfr.backends.hip.generator import HIPKernelGenerator
from pyfr.util import memoize


def get_grid_for_block(block, nrow, ncol=1):
    return (-(-nrow // block[0]), -(-ncol // block[1]), 1)


class HIPKernel(Kernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if hasattr(self, 'bind') and hasattr(self, 'add_to_graph'):
            self.gnodes = WeakKeyDictionary()


class HIPOrderedMetaKernel(MetaKernel):
    def add_to_graph(self, graph, dnodes):
        for k in self.kernels:
            dnodes = [k.add_to_graph(graph, dnodes)]

        return dnodes[0]


class HIPUnorderedMetaKernel(MetaKernel):
    def add_to_graph(self, graph, dnodes):
        nodes = [k.add_to_graph(graph, dnodes) for k in self.kernels]

        return graph.graph.add_empty(nodes)


class HIPKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes):
        return SourceModule(self.backend, src).get_function(name, argtypes)


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

    def _instantiate_kernel(self, dims, fun, arglst, argmv):
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

        return PointwiseKernel(*argmv)
