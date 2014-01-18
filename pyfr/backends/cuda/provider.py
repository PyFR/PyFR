# -*- coding: utf-8 -*-

from pycuda import compiler, driver

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, ComputeKernel)
import pyfr.backends.cuda.generator as generator
import pyfr.backends.cuda.types as types
from pyfr.util import memoize


def get_grid_for_block(block, nrow, ncol=1):
    return ((nrow + (-nrow % block[0])) // block[0],
            (ncol + (-ncol % block[1])) // block[1])


class CUDAKernelProvider(BaseKernelProvider):
    @memoize
    def _get_module(self, module, tplparams={}, nvccopts=None):
        # Get the template file
        tpl = self.backend.lookup.get_template(module)

        # Render the template
        mod = tpl.render(**tplparams)

        # Compile
        return compiler.SourceModule(mod, options=nvccopts)

    @memoize
    def _get_function(self, module, function, argtypes, tplparams={},
                      nvccopts=None):
        # Compile/retrieve the module
        mod = self._get_module(module, tplparams, nvccopts)

        # Get a reference to the function
        func = mod.get_function(function)

        # Prepare it for execution
        return func.prepare(argtypes)

    def _basic_kernel(self, fn, grid, block, *args):
        class BasicKernel(ComputeKernel):
            def run(self, scomp, scopy):
                fn.prepared_async_call(grid, block, scomp, *args)

        return BasicKernel()


class CUDAPointwiseKernelProvider(BasePointwiseKernelProvider):
    kernel_generator_cls = generator.CUDAKernelGenerator

    @memoize
    def _build_kernel(self, name, src, argtypes):
        # Ignore some spurious compiler warnings
        opts = ['-Xcudafe', '--diag_suppress=declared_but_not_referenced']

        # Compile the source code and retrieve the kernel
        fun = compiler.SourceModule(src, options=opts).get_function(name)

        # Prepare the kernel for execution
        fun.prepare(argtypes)

        # Declare a preference for L1 cache over shared memory
        fun.set_cache_config(driver.func_cache.PREFER_L1)

        return fun

    def _build_arglst(self, dims, argn, argt, argdict):
        # First arguments are the dimensions
        ndim, arglst = len(dims), list(dims)

        # Matrix types
        mattypes = (types.CUDAMatrixBank, types.CUDAMatrixBase)

        # Process non-dimensional arguments
        for aname, atypes in zip(argn[ndim:], argt[ndim:]):
            ka = argdict[aname]

            # Matrix
            if isinstance(ka, mattypes):
                arglst += [ka, ka.leadsubdim] if len(atypes) == 2 else [ka]
            # View
            elif isinstance(ka, (types.CUDAView, types.CUDAMPIView)):
                view = ka if isinstance(ka, types.CUDAView) else ka.view

                arglst += [view.basedata, view.mapping]
                arglst += [view.cstrides] if len(atypes) >= 3 else []
                arglst += [view.rstrides] if len(atypes) == 4 else []
            # Other; let PyCUDA handle it
            else:
                arglst.append(ka)

        return arglst

    def _instantiate_kernel(self, dims, fun, arglst):
        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, dims[-1])

        class PointwiseKernel(ComputeKernel):
            def run(self, scomp, scopy):
                fun.prepared_async_call(grid, block, scomp, *arglst)

        return PointwiseKernel()
