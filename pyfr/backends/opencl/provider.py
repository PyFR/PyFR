# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, ComputeKernel)
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
        cfg = self.backend.cfg

        # Determine the local work size
        if len(dims) == 1:
            ls = (cfg.getint('backend-opencl', 'local-size-1d', '64'),)
        else:
            ls = (cfg.getint('backend-opencl', 'local-size-2d', '128'),)

        # Global work size
        gs = tuple(gi - gi % -li for gi, li in zip(dims[::-1], ls))

        class PointwiseKernel(ComputeKernel):
            if any(isinstance(arg, str) for arg in arglst):
                def run(self, queue, **kwargs):
                    narglst = [kwargs.get(ka, ka) for ka in arglst]
                    narglst = [getattr(arg, 'data', arg) for arg in narglst]
                    fun(queue.cmd_q_comp, gs, ls, *narglst)
            else:
                def run(self, queue, **kwargs):
                    narglst = [getattr(arg, 'data', arg) for arg in arglst]
                    fun(queue.cmd_q_comp, gs, ls, *narglst)

        return PointwiseKernel()
