import numpy as np

from pyfr.backends.base.blasext import BaseBlasExtKernels
from pyfr.backends.metal.provider import MetalKernel, MetalKernelProvider
from pyfr.backends.metal.util import call_


class MetalBlasExtKernels(BaseBlasExtKernels, MetalKernelProvider):
    pvar_idx = 'VARIDX'

    def batched_inv(self, m):
        class BatchedInvKernel(MetalKernel):
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

        # Grid and threadgroup dimensions
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

    def zero(self, m):
        from Metal import NSMakeRange

        class ZeroKernel(MetalKernel):
            def run(self, cbuf):
                blit = cbuf.blitCommandEncoder()
                blit.fillBuffer_range_value_(m.basedata,
                                             NSMakeRange(m.offset, m.nbytes),
                                             0)
                blit.endEncoding()

        return ZeroKernel(mats=[m])

    def _reduction(self, fvvar, vvars, svars, tplargs):
        from Metal import MTLResourceStorageModeManaged

        ixdtype = self.backend.ixdtype
        nrow, _, ldim, fpdtype = fvvar.traits[1:]
        ncola, ncolb = fvvar.ioshape[-2:]
        nexprs = tplargs['nexprs']

        # Reduction threadgroup and grid dimensions
        blocksz = 128
        tgrp = (blocksz, 1, 1)
        grid = (ncolb - ncolb % -tgrp[0], ncola, 1)

        # Temporary buffer size requirements
        bufsz = fvvar.itemsize*nexprs*ncola*(grid[0] // tgrp[0])

        # Allocate the temporary buffer and map it on the host
        reduced_dev = call_(self.backend.dev, 'newBufferWith', length=bufsz,
                            options=MTLResourceStorageModeManaged)
        reduced_host = reduced_dev.contents().as_buffer(bufsz)
        reduced_host = np.frombuffer(reduced_host, dtype=fpdtype)
        reduced_host = reduced_host.reshape(nexprs, ncola, -1)

        # Add backend-specific template arguments
        tplargs['ncola'] = ncola
        tplargs['blocksz'] = blocksz

        # Get the kernel template
        src = self.backend.lookup.get_template('reduction').render(**tplargs)

        # Argument types for the reduction kernel
        argt = [ixdtype]*3 + [np.uintp]*(1 + len(vvars)) + [fpdtype]*len(svars)

        # Build the reduction kernel
        rkern = self._build_kernel('reduction', src, argt)

        # Constant scalar argument offset and count
        coff, nconsts = 4 + len(vvars), len(svars)

        # Kernel arguments
        kargs = [nrow, ncolb, ldim, (reduced_dev, 0)]
        kargs.extend(v.data for v in vvars.values())
        kargs.extend([None]*len(svars))

        # Reduction type
        reducer = np.max if tplargs['rop'] == 'max' else np.sum

        class ReductionKernel(MetalKernel):
            @property
            def retval(self):
                return reducer(reduced_host, axis=(1, 2))

            if svars:
                def bind(self, *consts):
                    kargs[coff:coff + nconsts] = consts

            def run(self, cbuf):
                rkern(cbuf, grid, tgrp, *kargs)
                blit = cbuf.blitCommandEncoder()
                blit.synchronizeResource_(reduced_dev)
                blit.endEncoding()

        return ReductionKernel(mats=vvars.values())
