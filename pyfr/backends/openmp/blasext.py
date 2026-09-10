import numpy as np

from pyfr.backends.base.blasext import BaseBlasExtKernels
from pyfr.backends.openmp.provider import OpenMPKernel, OpenMPKernelProvider


class OpenMPBlasExtKernels(BaseBlasExtKernels, OpenMPKernelProvider):
    pvar_idx = '_k'

    def _axnpby(self, arr, tplargs):
        nv, ixdtype = tplargs['nv'], self.backend.ixdtype
        nblocks, nrow, *_, fpdtype = arr[0].traits

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(**tplargs)

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [ixdtype] + [np.uintp]*nv + [fpdtype]*nv)

        # Set the static arguments
        kern.set_nblocks(nblocks)
        kern.set_args(nrow, *arr)

        class AxnpbyKernel(OpenMPKernel):
            def bind(self, *consts):
                self.kernel.set_args(*consts, start=1 + nv)

        return AxnpbyKernel(mats=arr, kernel=kern)

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

    def _reduction(self, fvvar, vvars, svars, tplargs):
        ixdtype = self.backend.ixdtype
        nblocks, nrow, *_, fpdtype = fvvar.traits
        narr = fvvar.ioshape[-1]
        ncola = fvvar.ioshape[-2] if len(fvvar.ioshape) >= 3 else 1

        # Add backend-specific template arguments
        tplargs['ncola'] = ncola

        # Render the reduction kernel template
        src = self.backend.lookup.get_template('reduction').render(**tplargs)

        # Array for the reduced data
        reduced = np.zeros(tplargs['nexprs'], dtype=fpdtype)

        # Argument types: ints, pointers (vvars), scalars (svars)
        argt = [ixdtype]*3 + [np.uintp]*(1 + len(vvars)) + [fpdtype]*len(svars)

        # Build and set arguments
        rkern = self._build_kernel('reduction', src, argt)
        args = [nrow, nblocks, narr, reduced.ctypes.data, *vvars.values()]
        rkern.set_args(*args)

        # Runtime argument offset for svars
        facoff = len(args)

        class ReductionKernel(OpenMPKernel):
            @property
            def retval(self):
                return reduced

            if svars:
                def bind(self, *consts):
                    self.kernel.set_args(*consts, start=facoff)

        return ReductionKernel(mats=vvars.values(), kernel=rkern)
