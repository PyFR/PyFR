from weakref import WeakKeyDictionary

import numpy as np

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

        if hasattr(self, 'add_to_graph'):
            self.gnodes = WeakKeyDictionary()


class HIPOrderedMetaKernel(BaseOrderedMetaKernel):
    def add_to_graph(self, graph, dnodes):
        for k in self.kernels:
            dnodes = [k.add_to_graph(graph, dnodes)]

        return dnodes[0]

class HIPUnorderedMetaKernel(BaseUnorderedMetaKernel):
    def add_to_graph(self, graph, dnodes):
        nodes = [k.add_to_graph(graph, dnodes) for k in self.kernels]

        return graph.graph.add_empty(nodes)


class HIPKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes):
        mod = HIPCompilerModule(self.backend, src)
        return mod.get_function(name, argtypes)

    def _benchmark(self, kfunc, nbench=4, nwarmup=1):
        try:
            stream = self._bench_stream
            start_evt = self._bench_start_evt
            stop_evt = self._bench_stop_evt
        except AttributeError:
            hip = self.backend.hip
            self._bench_stream = stream = hip.create_stream()
            self._bench_start_evt = start_evt = hip.create_event()
            self._bench_stop_evt = stop_evt = hip.create_event()

        for i in range(nbench + nwarmup):
            if i == nwarmup:
                start_evt.record(stream)

            kfunc(stream)

        stop_evt.record(stream)
        stream.synchronize()

        return stop_evt.elapsed_time(start_evt) / nbench

    def _bench_save(self, m):
        buf = np.empty(m.nbytes, dtype=np.uint8)
        self.backend.hip.memcpy(buf, m.data, m.nbytes)
        return buf

    def _bench_restore(self, m, buf):
        self.backend.hip.memcpy(m.data, buf, m.nbytes)

    def _bench_fill_random(self, m):
        blk = self._bench_rand_block(m.nbytes)
        for off in range(0, m.nbytes, blk.nbytes):
            self.backend.hip.memcpy(m.data + off, blk,
                                    min(blk.nbytes, m.nbytes - off))


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

    def _instantiate_kernel(self, dims, fun, args):
        block = self._block1d if len(dims) == 1 else self._block2d
        grid = get_grid_for_block(block, dims[-1])

        params = fun.make_params(grid, block)

        # Set the iteration dimensions
        params.set_args(*dims)

        class PointwiseKernel(HIPKernel):
            _set_arg = staticmethod(params.set_arg)

            def bind(self, **kwargs):
                super().bind(**kwargs)

                # Notify any graphs we're in about our new parameters
                for graph, gnode in self.gnodes.items():
                    graph.stale_kparams[gnode] = params

            def add_to_graph(self, graph, deps):
                gnode = graph.graph.add_kernel(params, deps)

                # Keep a (weak) graph reference so rebinds can notify it
                self.gnodes[graph] = gnode

                return gnode

            def run(self, stream):
                fun.exec_async(stream, params)

        return PointwiseKernel(args=args)
