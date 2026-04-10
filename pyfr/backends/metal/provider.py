from ctypes import sizeof

import numpy as np

from pyfr.backends.base import (BaseKernelProvider, BaseOrderedMetaKernel,
                                BasePointwiseKernelProvider,
                                BaseUnorderedMetaKernel, Kernel)
from pyfr.backends.metal.generator import MetalKernelGenerator
from pyfr.cache import memoize
from pyfr.nputil import npdtype_to_ctypestype


class MetalKernel(Kernel):
    def add_to_graph(self, graph, dnodes):
        graph.klist.append(self)

        return len(graph.klist)


class _MetalMetaKernel:
    def add_to_graph(self, graph, dnodes):
        for k in self.kernels:
            k.add_to_graph(graph, dnodes)

        return len(graph.klist)


class MetalOrderedMetaKernel(_MetalMetaKernel, BaseOrderedMetaKernel): pass
class MetalUnorderedMetaKernel(_MetalMetaKernel, BaseUnorderedMetaKernel): pass


class MetalKernelProvider(BaseKernelProvider):
    def _benchmark(self, kfunc, nbench=40, nwarmup=25):
        cbuf_warmup = self.backend.new_command_buffer()
        cbuf_bench = self.backend.new_command_buffer()

        for i in range(nwarmup):
            kfunc(cbuf_warmup)

        for i in range(nbench):
            kfunc(cbuf_bench)

        cbuf_warmup.commit()
        cbuf_bench.commit()
        cbuf_bench.waitUntilCompleted()

        return (cbuf_bench.GPUEndTime() - cbuf_bench.GPUStartTime()) / nbench

    @memoize
    def _build_kernel(self, name, src, argtypes, argn=[]):
        from Metal import MTLSizeMake

        # Build the pipeline using the compiler (with disk caching)
        cpsf, func = self.backend.compiler.build_pipeline(src, name)

        # Classify the arguments as either pointers or scalars
        pargs, sargs = [], []
        for i, argt in enumerate(argtypes):
            if argt == np.uintp:
                pargs.append(i)
            else:
                ctype = npdtype_to_ctypestype(argt)
                sargs.append((i, ctype(), sizeof(ctype)))

        def encode(cbuf, grid, tgrp, *args):
            cce = cbuf.computeCommandEncoder()
            cce.setComputePipelineState_(cpsf)

            for i in pargs:
                cce.setBuffer_offset_atIndex_(*args[i], i)

            for i, val, sz in sargs:
                val.value = args[i]
                cce.setBytes_length_atIndex_(val, sz, i)

            cce.dispatchThreads_threadsPerThreadgroup_(MTLSizeMake(*grid),
                                                       MTLSizeMake(*tgrp))
            cce.endEncoding()

        return encode


class MetalPointwiseKernelProvider(MetalKernelProvider,
                                   BasePointwiseKernelProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tgrp1d = (128, 1, 1)
        self._tgrp2d = (32, 8, 1)

        # Pass these block sizes to the generator
        class KernelGenerator(MetalKernelGenerator):
            block1d = self._tgrp1d
            block2d = self._tgrp2d

        self.kernel_generator_cls = KernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst, argm, argv):
        kargs, rtargs = [], []

        # Determine the thread group and grid sizes
        if len(dims) == 1:
            tgrp = self._tgrp1d
            grid = (dims[0] - dims[0] % -tgrp[0], 1, 1)
        else:
            tgrp = self._tgrp2d
            grid = (dims[1] - dims[1] % -tgrp[0], tgrp[1], 1)

        # Process the arguments
        for i, k in enumerate(arglst):
            if isinstance(k, str):
                kargs.append(None)
                rtargs.append((i, k))
            elif isinstance(k, (int, float)):
                kargs.append(k)
            else:
                k = getattr(k, 'data', k)
                kargs.append(k if isinstance(k, tuple) else (k, 0))

        class PointwiseKernel(MetalKernel):
            if rtargs:
                rtnames = tuple(k for _, k in rtargs)

                def bind(self, **kwargs):
                    for i, k in rtargs:
                        if k in kwargs:
                            kargs[i] = kwargs[k]

            def run(self, cbuf):
                fun(cbuf, grid, tgrp, *kargs)

        return PointwiseKernel(argm, argv)
