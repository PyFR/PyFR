from ctypes import Array, sizeof

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

    def _bench_view(self, m):
        buf = m.basedata.contents().as_buffer(m.offset + m.nbytes)
        return np.frombuffer(buf, dtype=np.uint8, offset=m.offset)

    def _bench_save(self, m):
        return self._bench_view(m).copy()

    def _bench_restore(self, m, buf):
        self._bench_view(m)[:] = buf

    def _bench_fill_random(self, m):
        v, blk = self._bench_view(m), self._bench_rand_block(m.nbytes)
        for off in range(0, m.nbytes, blk.nbytes):
            v[off:off + blk.nbytes] = blk[:m.nbytes - off]

    @memoize
    def _build_kernel(self, name, src, argtypes):
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
                if isinstance(val, Array):
                    val[:] = args[i]
                else:
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

    def _instantiate_kernel(self, dims, fun, args):
        # Determine the thread group and grid sizes
        if len(dims) == 1:
            tgrp = self._tgrp1d
            grid = (dims[0] - dims[0] % -tgrp[0], 1, 1)
        else:
            tgrp = self._tgrp2d
            grid = (dims[1] - dims[1] % -tgrp[0], tgrp[1], 1)

        # Argument setting with buffers as (buffer, offset) pairs
        def set_arg(i, k):
            match k:
                case int() | float():
                    kargs[i] = k
                case object(data=tuple() as v):
                    kargs[i] = v
                case object(data=v) | v:
                    kargs[i] = (v, 0)

        # Total argument count for the dimensions and named arguments
        nargs = len(dims) + sum(len(s[1]) for _, s, _ in args.values())

        # Set the iteration dimensions
        kargs = [None]*nargs
        for i, d in enumerate(dims):
            set_arg(i, int(d))

        class PointwiseKernel(MetalKernel):
            _set_arg = staticmethod(set_arg)

            def run(self, cbuf):
                fun(cbuf, grid, tgrp, *kargs)

        return PointwiseKernel(args=args)
