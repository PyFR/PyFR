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
            def run(self):
                dst.data[:] = src.data[:]

        return CopyKernel()
