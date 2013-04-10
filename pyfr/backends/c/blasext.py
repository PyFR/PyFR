# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.c.provider import CKernelProvider
from pyfr.backends.base import ComputeKernel
from pyfr.nputil import npdtype_to_ctype


class CBlasExtKernels(CKernelProvider):
    def axnpby(self, y, *xn):
        if any(y.traits != x.traits for x in xn):
            raise ValueError('Incompatible matrix types')

        opts = dict(n=len(xn), dtype=npdtype_to_ctype(y.dtype))
        fn = self._get_function('blasext', 'axnpby', None,
                                [np.uintp] + [np.uintp, y.dtype]*(1 + len(xn)),
                                opts)

        class AxnpbyKernel(ComputeKernel):
            def run(self, beta, *alphan):
                args = [i for axn in zip(xn, alphan) for i in axn]
                fn(y.leaddim*y.nrow, y, beta, *args)

        return AxnpbyKernel()
