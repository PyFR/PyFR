# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, Kernel)
from pyfr.backends.opencl.generator import OpenCLKernelGenerator
from pyfr.util import memoize


class OpenCLKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes):
        # Compile the source code
        prg = cl.Program(self.backend.ctx, src)
        prg.build(['-cl-fast-relaxed-math'])

        # Retrieve the kernel
        kern = getattr(prg, name)

        # Set the argument types
        dtypes = [t if t != np.intp else None for t in argtypes]
        kern.set_scalar_arg_dtypes(dtypes)

        return kern


class OpenCLPointwiseKernelProvider(OpenCLKernelProvider,
                                    BasePointwiseKernelProvider):
    kernel_generator_cls = OpenCLKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst):
        # Determine the work group sizes
        if len(dims) == 1:
            ls = (64,)
            gs = (dims[0] - dims[0] % -ls[0],)
        else:
            ls = (64, 4)
            gs = (dims[1] - dims[1] % -ls[0], ls[1])

        class PointwiseKernel(Kernel):
            if any(isinstance(arg, str) for arg in arglst):
                def run(self, queue, **kwargs):
                    narglst = [kwargs.get(ka, ka) for ka in arglst]
                    narglst = [getattr(arg, 'data', arg) for arg in narglst]
                    fun(queue.cmd_q, gs, ls, *narglst)
            else:
                def run(self, queue, **kwargs):
                    narglst = [getattr(arg, 'data', arg) for arg in arglst]
                    fun(queue.cmd_q, gs, ls, *narglst)

        return PointwiseKernel()
