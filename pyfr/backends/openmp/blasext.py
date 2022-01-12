# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.openmp.provider import OpenMPKernelProvider
from pyfr.backends.base import Kernel


class OpenMPBlasExtKernels(OpenMPKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        nblocks, nrow, *_, dtype = arr[0].traits
        ncola = arr[0].ioshape[-2]

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            subdims=subdims or range(ncola), ncola=ncola, nv=nv
        )

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [np.int32]*2 + [np.intp]*nv + [dtype]*nv)

        # Set the constant arguments
        kern.set_args(nrow, nblocks, *arr)

        class AxnpbyKernel(Kernel):
            def run(self, queue, *consts):
                kern.set_args(*consts, start=2 + nv)
                kern()

        return AxnpbyKernel(mats=arr)

    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        # Render the kernel template
        ksrc = self.backend.lookup.get_template('par-memcpy').render()

        dbbytes, sbbytes = dst.blocksz*dst.itemsize, src.blocksz*src.itemsize
        bnbytes = src.nrow*src.leaddim*src.itemsize
        nblocks = src.nblocks

        # Build the kernel
        kern = self._build_kernel('par_memcpy', ksrc, 'PiPiii')
        kern.set_args(dst, dbbytes, src, sbbytes, bnbytes, nblocks)

        class CopyKernel(Kernel):
            def run(self, queue):
                kern()

        return CopyKernel(mats=[dst, src])

    def reduction(self, *rs, method, norm, dt_mat=None):
        if any(r.traits != rs[0].traits for r in rs[1:]):
            raise ValueError('Incompatible matrix types')

        nblocks, nrow, *_, dtype = rs[0].traits
        ncola = rs[0].ioshape[-2]

        tplargs = dict(norm=norm, ncola=ncola, method=method)

        if method == 'resid':
            tplargs['dt_type'] = 'matrix' if dt_mat else 'scalar'

        # Render the reduction kernel template
        src = self.backend.lookup.get_template('reduction').render(**tplargs)

        # Array for the reduced data
        reduced = np.zeros(ncola, dtype=dtype)

        regs = list(rs) + [dt_mat] if dt_mat else rs

        # Argument types for reduction kernel
        if method == 'errest':
            argt = [np.int32]*2 + [np.intp]*4 + [dtype]*2
        elif method == 'resid' and dt_mat:
            argt = [np.int32]*2 + [np.intp]*4 + [dtype]
        else:
            argt = [np.int32]*2 + [np.intp]*3 + [dtype]

        # Build
        rkern = self._build_kernel('reduction', src, argt)
        rkern.set_args(nrow, nblocks, reduced.ctypes.data, *regs)

        # Runtime argument offset
        facoff = argt.index(dtype)

        class ReductionKernel(Kernel):
            @property
            def retval(self):
                return reduced

            def run(self, queue, *facs):
                rkern.set_args(*facs, start=facoff)
                rkern()

        return ReductionKernel(mats=regs)
