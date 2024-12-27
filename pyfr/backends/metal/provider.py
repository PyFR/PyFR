from ctypes import c_float, c_int32, c_int64, c_ulong, sizeof

import numpy as np

from pyfr.backends.base import (BaseKernelProvider, BaseOrderedMetaKernel,
                                BasePointwiseKernelProvider,
                                BaseUnorderedMetaKernel, Kernel)
from pyfr.backends.metal.util import call_
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
    typemap = [c_float, c_int32, c_int64, c_ulong]
    typemap = {k: (k(), sizeof(k)) for k in typemap}

    def _benchmark(self, kfunc, nbench=40, nwarmup=25):
        cbuf_warmup = self.backend.queue.commandBuffer()
        cbuf_bench = self.backend.queue.commandBuffer()

        for i in range(nwarmup):
            kfunc(cbuf_warmup)

        for i in range(nbench):
            kfunc(cbuf_bench)

        cbuf_warmup.commit()
        cbuf_bench.commit()
        cbuf_bench.waitUntilCompleted()

        return (cbuf_bench.GPUEndTime() - cbuf_bench.GPUStartTime()) / nbench

    @memoize
    def _build_program(self, src):
        from Metal import MTLCompileOptions

        # Set the compiler options
        opts = MTLCompileOptions.new()
        opts.setFastMathEnabled_(True)

        # Compile the kernel
        lib, err = call_(self.backend.dev, 'newLibraryWith', source=src,
                         options=opts, error=None)
        if err is not None:
            raise ValueError(f'Compiler error: {err}')

        return lib

    def _build_kernel(self, name, src, argtypes, argn=[]):
        from Metal import MTLComputePipelineDescriptor, MTLSizeMake

        # Build the program
        lib = self._build_program(src)

        # Fetch the function
        func = call_(lib, 'newFunctionWith', name=name)
        if func is None:
            raise KeyError(f'Unable to load function {name}')

        # Create the pipeline descriptor
        desc = MTLComputePipelineDescriptor.alloc().init()
        desc.setComputeFunction_(func)
        desc.setThreadGroupSizeIsMultipleOfThreadExecutionWidth_(True)

        # Obtain the corresponding compute pipeline
        cpsf = call_(self.backend.dev, 'newComputePipelineStateWith',
                     descriptor=desc, error=None)
        if cpsf is None:
            raise RuntimeError('Unable to create compute pipeline state')

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
                buf, off = args[i]
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
                def bind(self, **kwargs):
                    for i, k in rtargs:
                        kargs[i] = kwargs[k]

            def run(self, cbuf):
                fun(cbuf, grid, tgrp, *kargs)

        return PointwiseKernel(argm, argv)
