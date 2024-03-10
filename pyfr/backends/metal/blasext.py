import numpy as np

from pyfr.backends.metal.provider import MetalKernel, MetalKernelProvider
from pyfr.backends.metal.util import call_


class MetalBlasExtKernels(MetalKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            subdims=subdims or range(ncola), ncola=ncola, nv=nv
        )

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [ixdtype]*2 + [np.uintp]*nv + [fpdtype]*nv)
        grid, tgrp = (ncolb, nrow, 1), (128, 1, 1)
        kargs = [ncolb, ldim] + [a.data for a in arr] + [1.0]*nv

        class AxnpbyKernel(MetalKernel):
            def bind(self, *consts):
                kargs[2 + nv:] = consts

            def run(self, cbuf):
                kern(cbuf, grid, tgrp, *kargs)

        return AxnpbyKernel(mats=arr)

    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(MetalKernel):
            def run(self, cbuf):
                blit = cbuf.blitCommandEncoder()
                call_(blit, 'copy', fromBuffer=src.basedata,
                      sourceOffset=src.offset, toBuffer=dst.basedata,
                      destinationOffset=dst.offset, size=dst.nbytes)
                blit.endEncoding()

        return CopyKernel(mats=[dst, src])

    def reduction(self, *rs, method, norm, dt_mat=None):
        from Metal import MTLResourceStorageModeManaged

        if any(r.traits != rs[0].traits for r in rs[1:]):
            raise ValueError('Incompatible matrix types')

        ixdtype = self.backend.ixdtype
        nrow, ncol, ldim, fpdtype = rs[0].traits[1:]
        ncola, ncolb = rs[0].ioshape[1:]
        itemsize = rs[0].itemsize

        # Reduction threadgroup and grid dimensions
        tgrp = (128, 1, 1)
        grid = (ncolb - ncolb % -tgrp[0], ncola, 1)

        # Temporary buffer size requirements
        bufsz = itemsize*ncola*(grid[0] // tgrp[0])

        # Allocate the temporary buffer and map it on the host
        reduced_dev = call_(self.backend.dev, 'newBufferWith', length=bufsz,
                            options=MTLResourceStorageModeManaged)
        reduced_host = reduced_dev.contents().as_buffer(bufsz)
        reduced_host = np.frombuffer(reduced_host, dtype=fpdtype)
        reduced_host = reduced_host.reshape(ncola, -1)

        # Template arguments
        tplargs = dict(blocksz=tgrp[0], ncola=ncola, norm=norm, method=method)

        if method == 'resid':
            tplargs['dt_type'] = 'matrix' if dt_mat else 'scalar'

        # Get the kernel template
        src = self.backend.lookup.get_template('reduction').render(**tplargs)

        regs = [*rs, dt_mat] if dt_mat else rs

        # Argument types for the reduction kernel
        if method == 'errest':
            argt = [ixdtype]*3 + [np.uintp]*4 + [fpdtype]*2
        elif method == 'resid' and dt_mat:
            argt = [ixdtype]*3 + [np.uintp]*4 + [fpdtype]
        else:
            argt = [ixdtype]*3 + [np.uintp]*3 + [fpdtype]

        # Build the reduction kernel
        rkern = self._build_kernel('reduction', src, argt)

        # Runtime argument offset
        facoff = argt.index(fpdtype)
        nfacs = 2 if method == 'errest' else 1

        # Kernel arguments
        kargs = [nrow, ncolb, ldim, (reduced_dev, 0)]
        kargs.extend(r.data for r in regs)
        kargs.extend([None]*nfacs)

        # Norm type
        reducer = np.max if norm == 'uniform' else np.sum

        class ReductionKernel(MetalKernel):
            @property
            def retval(self):
                return reducer(reduced_host, axis=1)

            def bind(self, *facs):
                kargs[facoff:facoff + nfacs] = facs

            def run(self, cbuf):
                rkern(cbuf, grid, tgrp, *kargs)
                blit = cbuf.blitCommandEncoder()
                blit.synchronizeResource_(reduced_dev)
                blit.endEncoding()

        return ReductionKernel(mats=regs)
