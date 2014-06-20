# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, ComputeKernel)
import pyfr.backends.opencl.generator as generator
from pyfr.util import memoize


class OpenCLKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes):
        # Compile the source code
        prg = cl.Program(self.backend.ctx, bytes(src))
        prg.build(['-cl-fast-relaxed-math'])

        # Retrieve the kernel
        kern = getattr(prg, name)

        # Set the argument types
        dtypes = [t if t != np.intp else None for t in argtypes]
        kern.set_scalar_arg_dtypes(dtypes)

        return kern


class OpenCLPointwiseKernelProvider(OpenCLKernelProvider,
                                    BasePointwiseKernelProvider):
    kernel_generator_cls = generator.OpenCLKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst):
        class PointwiseKernel(ComputeKernel):
            def run(self, queue):
                narglst = [getattr(arg, 'data', arg) for arg in arglst]
                fun(queue.cl_queue_comp, (dims[-1],), None, *narglst)

        return PointwiseKernel()
