# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

from pyfr.backends.opencl.provider import OpenCLKernelProvider
from pyfr.backends.base import ComputeKernel


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

        class AxnpbyKernel(ComputeKernel):
            def run(self, queue, *consts):
                arrd = [x.data for x in arr]
                kern(queue.cmd_q_comp, (ncolb, nrow), None, nrow, ncolb, ldim,
                     *arrd, *consts)

        return AxnpbyKernel()

    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(ComputeKernel):
            def run(self, queue):
                cl.enqueue_copy(queue.cmd_q_comp, dst.data, src.data)

        return CopyKernel()

    def reduction(self, *rs, method, norm, dt_mat=None):
        if any(r.traits != rs[0].traits for r in rs[1:]):
            raise ValueError('Incompatible matrix types')

        nrow, ncol, ldim, dtype = rs[0].traits[1:]
        ncola, ncolb = rs[0].ioshape[1:]

        # Reduction workgroup dimensions
        ls = (128, 1)
        gs = (ncolb - ncolb % -ls[0], ncola)

        # Empty result buffer on host with (nvars, ngroups)
        reduced_host = np.empty((ncola, gs[0] // ls[0]), dtype)

        # Device memory allocation
        reduced_dev = cl.Buffer(self.backend.ctx, cl.mem_flags.READ_WRITE,
                                reduced_host.nbytes)

        tplargs = dict(norm=norm, sharesz=ls[0], method=method)

        if method == 'resid':
            tplargs['dt_type'] = 'matrix' if dt_mat else 'scalar'

        # Get the kernel template
        src = self.backend.lookup.get_template('reduction').render(**tplargs)

        rdata = [r.data for r in rs]
        rdata += [dt_mat.data] if dt_mat else []

        # Argument types for reduction kernel
        if method == 'errest':
            argt = [np.int32]*3 + [np.intp]*4 + [dtype]*2
        elif method == 'resid' and dt_mat:
            argt = [np.int32]*3 + [np.intp]*4 + [dtype]
        else:
            argt = [np.int32]*3 + [np.intp]*3 + [dtype]

        # Build the reduction kernel
        rkern = self._build_kernel('reduction', src, argt)

        # Norm type
        reducer = np.max if norm == 'uniform' else np.sum

        class ReductionKernel(ComputeKernel):
            @property
            def retval(self):
                return reducer(reduced_host, axis=1)

            def run(self, queue, *facs):
                rkern(queue.cmd_q_comp, gs, ls,
                      nrow, ncolb, ldim, reduced_dev, *rdata, *facs)
                cevent = cl.enqueue_copy(queue.cmd_q_comp, reduced_host,
                                         reduced_dev, is_blocking=False)
                queue.copy_events.append(cevent)

        return ReductionKernel()
