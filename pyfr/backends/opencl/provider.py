# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, Kernel)
from pyfr.backends.opencl.generator import OpenCLKernelGenerator
from pyfr.nputil import npdtype_to_ctypestype
from pyfr.util import memoize


class OpenCLKernelProvider(BaseKernelProvider):
    @memoize
    def _build_program(self, src):
        flags = ['-cl-fast-relaxed-math', '-cl-std=CL2.0']

        return self.backend.cl.program(src, flags)

    def _build_kernel(self, name, src, argtypes):
        argtypes = [npdtype_to_ctypestype(arg) for arg in argtypes]

        return self._build_program(src).get_kernel(name, argtypes)


class OpenCLPointwiseKernelProvider(OpenCLKernelProvider,
                                    BasePointwiseKernelProvider):
    kernel_generator_cls = OpenCLKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst, argmv):
        rtargs = []

        # Determine the work group sizes
        if len(dims) == 1:
            ls = (64,)
            gs = (dims[0] - dims[0] % -ls[0],)
        else:
            ls = (64, 4)
            gs = (dims[1] - dims[1] % -ls[0], ls[1])

        # Process the arguments
        for i, k in enumerate(arglst):
            if isinstance(k, str):
                rtargs.append((i, k))
            else:
                fun.set_arg(i, k)

        class PointwiseKernel(Kernel):
            def run(self, queue, **kwargs):
                for i, k in rtargs:
                    fun.set_arg(i, kwargs[k])

                fun.exec_async(queue.cmd_q, gs, ls)

        return PointwiseKernel(*argmv)
