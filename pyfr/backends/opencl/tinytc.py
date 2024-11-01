from ctypes import (POINTER, byref, c_double, c_int, c_int32, c_int64, c_float,
                    c_size_t, c_uint, c_void_p)

import numpy as np

from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider
from pyfr.ctypesutil import LibWrapper


# Possible TinyTC exception types
class TinyTCError(Exception): pass


class TinyTCWrappers(LibWrapper):
    _libname = 'tinytc_cl'

    # Error codes
    _statuses = {
        '*': TinyTCError
    }

    # Constants
    MEM_TYPE_BUF = 0
    SCALAR_TYPE_F32 = 6
    SCALAR_TYPE_F64 = 7

    # Functions
    _functions = [
        (c_int, 'tinytc_cl_get_support_level', c_void_p, POINTER(c_int)),
        (c_int, 'tinytc_cl_core_info_create', POINTER(c_void_p), c_void_p),
        (c_int, 'tinytc_cl_recipe_handler_submit', c_void_p, c_void_p,
         c_uint, POINTER(c_void_p), POINTER(c_void_p)),
        (c_int, 'tinytc_cl_recipe_handler_create', POINTER(c_void_p),
         c_void_p, c_void_p, c_void_p, c_void_p),
        (c_int, 'tinytc_recipe_tall_and_skinny_create', POINTER(c_void_p),
         c_void_p, c_int, c_int64, c_int64, c_int32, c_void_p),
        (c_int, 'tinytc_recipe_tall_and_skinny_set_args', c_void_p,
         c_int64, c_size_t, c_void_p, c_int, c_void_p, c_int64,
         c_int, c_void_p, c_int64, c_size_t, c_void_p, c_int, c_int64,
         c_int64),
        (c_int, 'tinytc_recipe_handler_release', c_void_p),
        (c_int, 'tinytc_recipe_release', c_void_p),
        (c_int, 'tinytc_core_info_release', c_void_p)
    ]


class OpenCLTinyTCKernels(OpenCLKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        self.handle = c_void_p()

        # Recipe and timing caches
        self._recipes = {}
        self._mul_timing = {}

        # Load and wrap TinyTC
        self.lib = TinyTCWrappers()

        # Query the support level
        support = c_int()
        self.lib.tinytc_cl_get_support_level(backend.cl.dev, support)

        if support:
            # Init
            self.lib.tinytc_cl_core_info_create(self.handle, backend.cl.dev)

            self.mul = self._mul

    def __del__(self):
        for r in self._recipes.values():
            self.lib.tinytc_recipe_release(r)

        if self.handle:
            self.lib.tinytc_core_info_release(self.handle)

    def _get_tall_and_skinny_recipe(self, stype, n, k):
        try:
            return self._recipes[stype, n, k]
        except KeyError:
            recipe = c_void_p()
            self.lib.tinytc_recipe_tall_and_skinny_create(recipe, self.handle,
                                                          stype, n, k, 0, None)

            self._recipes[stype, n, k] = recipe

            return recipe

    def _mul(self, a, b, out, alpha=1.0, beta=0.0):
        cl = self.backend.cl
        w = self.lib

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        m, n, k = b.ncol, a.nrow, a.ncol
        A, B, C = b, a, out

        if a.dtype == np.float64:
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
            csize = 8
            stype = w.SCALAR_TYPE_F64
        else:
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)
            csize = 4
            stype = w.SCALAR_TYPE_F32

        # Create a tall-and-skinny recipe
        recipe = self._get_tall_and_skinny_recipe(stype, n, k)

        # Cache key
        ckey = (a.dtype, alpha, beta, m, n, k, a.leaddim, b.leaddim,
                out.leaddim)

        # Create the associated handler
        handler = c_void_p()
        w.tinytc_cl_recipe_handler_create(handler, cl.ctx, cl.dev, recipe,
                                          None)

        try:
            # Set the arguments
            w.tinytc_recipe_tall_and_skinny_set_args(
                handler, m, csize, byref(alpha_ct), w.MEM_TYPE_BUF, A,
                A.leaddim, w.MEM_TYPE_BUF, B, B.leaddim, csize, byref(beta_ct),
                w.MEM_TYPE_BUF, C, C.leaddim
            )

            # Obtain the performance of the kernel
            try:
                dt = self._mul_timing[ckey]
            except KeyError:
                # Save a copy of the contents of the output matrix
                out_np = getattr(out, 'parent', out).get()

                def gemm(queue):
                    evt_ptr = c_void_p()

                    w.tinytc_cl_recipe_handler_submit(handler, queue, 0, None,
                                                      evt_ptr)

                    return cl.event(evt_ptr)

                # Benchmark the kernel and update the cache
                self._mul_timing[ckey] = dt = self._benchmark(gemm)

                # Restore the output matrix
                getattr(out, 'parent', out).set(out_np)
        except:
            w.tinytc_recipe_handler_release(handler)
            raise

        class MulKernel(OpenCLKernel):
            def __del__(self):
                w.tinytc_recipe_handler_release(handler)

            def run(self, queue, wait_for=None, ret_evt=False):
                evt_ptr = c_void_p() if ret_evt else None

                if wait_for:
                    nwait = len(wait_for)
                    wait_e = (c_void_p * nwait)(*[int(e) for e in wait_for])
                else:
                    nwait = 0
                    wait_e = None

                w.tinytc_cl_recipe_handler_submit(handler, queue, nwait,
                                                  wait_e, evt_ptr)

                if ret_evt:
                    return cl.event(evt_ptr)

        return MulKernel(mats=[a, b, out], dt=dt)
