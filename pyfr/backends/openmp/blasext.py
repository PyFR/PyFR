# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.openmp.provider import OpenMPKernelProvider
from pyfr.backends.base import ComputeKernel


class OpenMPBlasExtKernels(OpenMPKernelProvider):
    def axnpby(self, y, *xn):
        if any(y.traits != x.traits for x in xn):
            raise ValueError('Incompatible matrix types')

        nv, cnt = len(xn), y.leaddim*y.nrow

        # Render the kernel template
        tpl = self.backend.lookup.get_template('axnpby')
        src = tpl.render(n=nv, alignb=self.backend.alignb, fpdtype=y.dtype)

        # Build
        kern = self._build_kernel('axnpby', src,
                                  [np.int32] + [np.intp, y.dtype]*(1 + nv))

        class AxnpbyKernel(ComputeKernel):
            def run(self, queue, beta, *alphan):
                args = [i for axn in zip(xn, alphan) for i in axn]
                kern(cnt, y, beta, *args)

        return AxnpbyKernel()

    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(ComputeKernel):
            def run(self, queue):
                dst.data[:] = src.data[:]

        return CopyKernel()

    def errest(self, x, y, z):
        if x.traits != y.traits != z.traits:
            raise ValueError('Incompatible matrix types')

        cnt = x.leaddim*x.nrow
        dtype = x.dtype

        # Render the reduction kernel template
        tpl = self.backend.lookup.get_template('errest')
        src = tpl.render(alignb=self.backend.alignb, fpdtype=dtype)

        # Build
        rkern = self._build_kernel(
            'errest', src, [np.int32] + [np.intp]*3 + [dtype]*2, restype=dtype
        )

        class ErrestKernel(ComputeKernel):
            @property
            def retval(self):
                return self._retval

            def run(self, queue, atol, rtol):
                self._retval = rkern(cnt, x, y, z, atol, rtol)

        return ErrestKernel()
