import numpy as np

from pyfr.backends.base.blasext import BaseBlasExtKernels
from pyfr.backends.openmp.provider import OpenMPKernel, OpenMPKernelProvider


class OpenMPBlasExtKernels(BaseBlasExtKernels, OpenMPKernelProvider):
    # Reduce with a two dimensional kernel
    reduction_ndim = 2

    def copy(self, dst, src):
        ixdtype = self.backend.ixdtype

        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        # Render the kernel template
        ksrc = self.backend.lookup.get_template('par-memop').render(op='copy')

        dbbytes, sbbytes = dst.blocksz*dst.itemsize, src.blocksz*src.itemsize
        bnbytes = src.nrow*src.leaddim*src.itemsize
        nblocks = src.nblocks

        # Build the kernel
        kern = self._build_kernel('par_copy', ksrc,
                                  [np.uintp]*2 + [ixdtype]*4)
        kern.set_args(dst, src, dbbytes, sbbytes, bnbytes, nblocks)

        return OpenMPKernel(mats=[dst, src], kernel=kern)

    def zero(self, m):
        ixdtype = self.backend.ixdtype

        # Render the kernel template
        ksrc = self.backend.lookup.get_template('par-memop').render(op='zero')

        dbbytes = m.blocksz*m.itemsize
        bnbytes = m.nrow*m.leaddim*m.itemsize
        nblocks = m.nblocks

        # Build the kernel
        kern = self._build_kernel('par_zero', ksrc,
                                  [np.uintp] + [ixdtype]*3)
        kern.set_args(m, dbbytes, bnbytes, nblocks)

        return OpenMPKernel(mats=[m], kernel=kern)
