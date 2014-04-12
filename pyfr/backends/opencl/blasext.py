# -*- coding: utf-8 -*-

import numpy as np
from pyopencl.array import splay

from pyfr.backends.opencl.provider import OpenCLKernelProvider
from pyfr.backends.base import ComputeKernel


class OpenCLBlasExtKernels(OpenCLKernelProvider):
    def axnpby(self, *arr):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        nrow, leaddim, leadsubdim, dtype = arr[0].traits

        # Render the kernel template
        tpl = self.backend.lookup.get_template('axnpby')
        src = tpl.render(nv=nv, alignb=self.backend.alignb, fpdtype=dtype)

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [np.int32] + [np.intp]*nv + [dtype]*nv)

        # Determine the total element count in the matrices
        cnt = leaddim*nrow

        # Compute a suitable global and local workgroup sizes
        gs, ls = splay(self.backend.qdflt, cnt)

        class AxnpbyKernel(ComputeKernel):
            def run(self, qcomp, qcopy, *consts):
                args = [x.data for x in arr] + list(consts)
                kern(qcomp, gs, ls, cnt, *args)

        return AxnpbyKernel()
