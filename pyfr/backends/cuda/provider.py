# -*- coding: utf-8 -*-

import pycuda.compiler as compiler

from pyfr.backends.base import ComputeKernel
from pyfr.util import memoize


def get_grid_for_block(block, nrow, ncol=1):
    return ((nrow + (-nrow % block[0])) // block[0],
            (ncol + (-ncol % block[1])) // block[1])


def get_2d_grid_block(function, nrow, ncol):
    # TODO: Write a totally bitchin' method which uses info from the
    #       function to help compute an optimal block size
    block = (min(16, nrow), min(16, ncol), 1)
    grid = get_grid_for_block(block, nrow, ncol)
    return grid, block


class CUDAKernelProvider(object):
    def __init__(self, backend):
        self.backend = backend

    @memoize
    def _get_module(self, module, tplparams={}, nvccopts=None):
        # Get the template file
        tpl = self.backend.lookup.get_template(module)

        # Filter floating point constants
        if tplparams['dtype'] == 'float':
            fpfilt = lambda v: v + ('f' if '.' in v or 'e' in v else '.f')
        else:
            fpfilt = lambda v: v

        # Render the template
        mod = tpl.render(f=fpfilt, **tplparams)

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
