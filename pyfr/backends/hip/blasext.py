import numpy as np

from pyfr.backends.hip.provider import (HIPKernel, HIPKernelProvider,
                                        get_grid_for_block)


class HIPBlasExtKernels(HIPKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb, nrow)

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            block=block, subdims=subdims or range(ncola), ncola=ncola, nv=nv
        )

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [ixdtype]*2 + [np.uintp]*nv + [fpdtype]*nv)

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(ncolb, ldim, *arr)

        class AxnpbyKernel(HIPKernel):
            def bind(self, *consts):
                params.set_args(*consts, start=2 + nv)

            def run(self, stream):
                kern.exec_async(stream, params)

        return AxnpbyKernel(mats=arr)

    def copy(self, dst, src):
        hip = self.backend.hip

        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(HIPKernel):
            def add_to_graph(self, graph, deps):
                return graph.graph.add_memcpy(dst, src, dst.nbytes, deps)

            def run(self, stream):
                hip.memcpy(dst, src, dst.nbytes, stream)

        return CopyKernel(mats=[dst, src])

    def reduction(self, *rs, method, norm, dt_mat=None):
        if any(r.traits != rs[0].traits for r in rs[1:]):
            raise ValueError('Incompatible matrix types')

        hip = self.backend.hip
        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = rs[0].traits[1:]
        ncola, ncolb = rs[0].ioshape[1:]

        # Reduction block dimensions
        block = (128, 1, 1)

        # Determine the grid size
        grid = get_grid_for_block(block, ncolb, ncola)

        # Empty result buffer on the device
        reduced_dev = hip.mem_alloc(ncola*grid[0]*rs[0].itemsize)

        # Empty result buffer on the host
        reduced_host = hip.pagelocked_empty((ncola, grid[0]), fpdtype)

        tplargs = dict(norm=norm, blocksz=block[0], method=method)

        if method == 'resid':
            tplargs['dt_type'] = 'matrix' if dt_mat else 'scalar'

        # Get the kernel template
        src = self.backend.lookup.get_template('reduction').render(**tplargs)

        regs = list(rs) + [dt_mat] if dt_mat else rs

        # Argument types for reduction kernel
        if method == 'errest':
            argt = [ixdtype]*3 + [np.uintp]*4 + [fpdtype]*2
        elif method == 'resid' and dt_mat:
            argt = [ixdtype]*3 + [np.uintp]*4 + [fpdtype]
        else:
            argt = [ixdtype]*3 + [np.uintp]*3 + [fpdtype]

        # Build the reduction kernel
        rkern = self._build_kernel('reduction', src, argt)

        # Set the parameters
        params = rkern.make_params(grid, block)
        params.set_args(nrow, ncolb, ldim, reduced_dev, *regs)

        # Runtime argument offset
        facoff = argt.index(fpdtype)

        # Norm type
        reducer = np.max if norm == 'uniform' else np.sum

        class ReductionKernel(HIPKernel):
            @property
            def retval(self):
                return reducer(reduced_host, axis=1)

            def bind(self, *facs):
                params.set_args(*facs, start=facoff)

            def run(self, stream):
                rkern.exec_async(stream, params)
                hip.memcpy(reduced_host, reduced_dev, reduced_dev.nbytes,
                           stream)

        return ReductionKernel(mats=regs)
