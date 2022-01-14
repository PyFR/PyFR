# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.opencl.provider import OpenCLKernelProvider
from pyfr.backends.base import Kernel


class OpenCLBlasExtKernels(OpenCLKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        nrow, ncol, ldim, dtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            subdims=subdims or range(ncola), ncola=ncola, nv=nv
        )

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [np.int32]*3 + [np.intp]*nv + [dtype]*nv)
        kern.set_args(nrow, ncolb, ldim, *arr)

        class AxnpbyKernel(Kernel):
            def run(self, queue, *consts):
                kern.set_args(*consts, start=3 + nv)
                kern.exec_async(queue.cmd_q, (ncolb, nrow), None)

        return AxnpbyKernel(mats=arr)

    def copy(self, dst, src):
        cl = self.backend.cl

        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(Kernel):
            def run(self, queue):
                cl.memcpy_async(queue.cmd_q, dst, src, dst.nbytes)

        return CopyKernel()

    def reduction(self, *rs, method, norm, dt_mat=None):
        if any(r.traits != rs[0].traits for r in rs[1:]):
            raise ValueError('Incompatible matrix types')

        cl = self.backend.cl
        nrow, ncol, ldim, dtype = rs[0].traits[1:]
        ncola, ncolb = rs[0].ioshape[1:]

        # Reduction workgroup dimensions
        ls = (128, 1)
        gs = (ncolb - ncolb % -ls[0], ncola)

        # Empty result buffer on host with (nvars, ngroups)
        reduced_host = np.empty((ncola, gs[0] // ls[0]), dtype)

        # Corresponding device memory allocation
        reduced_dev = cl.mem_alloc(reduced_host.nbytes)

        tplargs = dict(norm=norm, sharesz=ls[0], method=method)

        if method == 'resid':
            tplargs['dt_type'] = 'matrix' if dt_mat else 'scalar'

        # Get the kernel template
        src = self.backend.lookup.get_template('reduction').render(**tplargs)

        regs = list(rs) + [dt_mat] if dt_mat else rs

        # Argument types for reduction kernel
        if method == 'errest':
            argt = [np.int32]*3 + [np.intp]*4 + [dtype]*2
        elif method == 'resid' and dt_mat:
            argt = [np.int32]*3 + [np.intp]*4 + [dtype]
        else:
            argt = [np.int32]*3 + [np.intp]*3 + [dtype]

        # Build the reduction kernel
        rkern = self._build_kernel('reduction', src, argt)
        rkern.set_args(nrow, ncolb, ldim, reduced_dev, *regs)

        # Norm type
        reducer = np.max if norm == 'uniform' else np.sum

        # Runtime argument offset
        facoff = argt.index(dtype)

        class ReductionKernel(Kernel):
            @property
            def retval(self):
                return reducer(reduced_host, axis=1)

            def run(self, queue, *facs):
                rkern.set_args(*facs, start=facoff)
                rkern.exec_async(queue.cmd_q, gs, ls)
                cl.memcpy_async(queue.cmd_q, reduced_host, reduced_dev,
                                reduced_dev.nbytes)

        return ReductionKernel(mats=regs)
