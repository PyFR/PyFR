import numpy as np

from pyfr.backends.base.blasext import BaseBlasExtKernels
from pyfr.backends.cuda.provider import (CUDAKernel, CUDAKernelProvider,
                                         get_grid_for_block)


class CUDABlasExtKernels(BaseBlasExtKernels, CUDAKernelProvider):
    pvar_idx = 'VARIDX'

    def batched_inv(self, m):
        class BatchedInvKernel(CUDAKernel):
            def run(self, stream):
                M = m.get().transpose(2, 0, 1)
                m.set(np.linalg.inv(M).transpose(1, 2, 0))

        return BatchedInvKernel(mats=[m])

    def _axnpby(self, arr, tplargs):
        nv, ixdtype = tplargs['nv'], self.backend.ixdtype
        nrow, _, ldim, fpdtype = arr[0].traits[1:]
        ncolb = arr[0].ioshape[-1]

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(**tplargs)

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [ixdtype]*2 + [np.uintp]*nv + [fpdtype]*nv)

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb, nrow)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(ncolb, ldim, *arr)

        class AxnpbyKernel(CUDAKernel):
            def bind(self, *consts):
                params.set_args(*consts, start=2 + nv)

            def run(self, stream):
                kern.exec_async(stream, params)

        return AxnpbyKernel(mats=arr)

    def copy(self, dst, src):
        cuda = self.backend.cuda

        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(CUDAKernel):
            def add_to_graph(self, graph, deps):
                return graph.graph.add_memcpy(dst, src, dst.nbytes, deps)

            def run(self, stream):
                cuda.memcpy(dst, src, dst.nbytes, stream)

        return CopyKernel(mats=[dst, src])

    def zero(self, m):
        cuda = self.backend.cuda

        class ZeroKernel(CUDAKernel):
            def add_to_graph(self, graph, deps):
                return graph.graph.add_memset(m, 0, m.nbytes, deps)

            def run(self, stream):
                cuda.memset(m, 0, m.nbytes, stream)

        return ZeroKernel(mats=[m])

    def _reduction(self, fvvar, vvars, svars, tplargs):
        cuda = self.backend.cuda
        ixdtype = self.backend.ixdtype
        nrow, _, ldim, fpdtype = fvvar.traits[1:]
        ncola, ncolb = fvvar.ioshape[1:]
        nexprs = tplargs['nexprs']

        # Reduction block dimensions (use more blocks for atomic approach)
        block = (256, 1, 1)
        nblocks = min(1024, (ncolb + block[0] - 1) // block[0])
        grid = (nblocks, ncola, 1)

        # Result buffer on device (nexprs*ncola, not nexprs*ncola*nblocks)
        reduced_dev = cuda.mem_alloc(nexprs*ncola*fvvar.itemsize)

        # Result buffer on host
        reduced_host = cuda.pagelocked_empty((nexprs, ncola), fpdtype)

        # Initialisation buffer (0 for sum, -fpdtype_max for max)
        init_host = cuda.pagelocked_empty((nexprs, ncola), fpdtype)
        init_host.fill(tplargs['init_val'])

        # Add backend-specific template arguments
        tplargs['ncola'] = ncola

        # Get the kernel template
        src = self.backend.lookup.get_template('reduction').render(**tplargs)

        # Argument types for the reduction kernel
        argt = [ixdtype]*3 + [np.uintp]*(1 + len(vvars)) + [fpdtype]*len(svars)

        # Build the reduction kernel
        rkern = self._build_kernel('reduction', src, argt)

        # Set the parameters
        params = rkern.make_params(grid, block)
        params.set_args(nrow, ncolb, ldim, reduced_dev, *vvars.values())

        # Runtime argument offset
        coff = 4 + len(vvars)

        # Host-side reduction over ncola dimension
        reducer = np.max if tplargs['rop'] == 'max' else np.sum

        class ReductionKernel(CUDAKernel):
            @property
            def retval(self):
                return reducer(reduced_host, axis=1)

            if svars:
                def bind(self, *consts):
                    params.set_args(*consts, start=coff)

            def run(self, stream):
                cuda.memcpy(reduced_dev, init_host, reduced_dev.nbytes, stream)
                rkern.exec_async(stream, params)
                cuda.memcpy(reduced_host, reduced_dev, reduced_dev.nbytes,
                            stream)

        return ReductionKernel(mats=vvars.values())
