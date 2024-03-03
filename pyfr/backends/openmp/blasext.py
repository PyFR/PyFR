import numpy as np

from pyfr.backends.openmp.provider import OpenMPKernel, OpenMPKernelProvider


class OpenMPBlasExtKernels(OpenMPKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        ixdtype = self.backend.ixdtype
        nblocks, nrow, *_, fpdtype = arr[0].traits
        ncola = arr[0].ioshape[-2]

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            subdims=subdims or range(ncola), ncola=ncola, nv=nv
        )

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
        ksrc = self.backend.lookup.get_template('par-memcpy').render()

        dbbytes, sbbytes = dst.blocksz*dst.itemsize, src.blocksz*src.itemsize
        bnbytes = src.nrow*src.leaddim*src.itemsize
        nblocks = src.nblocks

        # Build the kernel
        kern = self._build_kernel('par_memcpy', ksrc,
                                  [np.uintp]*2 + [ixdtype]*4)
        kern.set_args(dst, src, dbbytes, sbbytes, bnbytes, nblocks)

        return OpenMPKernel(mats=[dst, src], kernel=kern)

    def reduction(self, *rs, method, norm, dt_mat=None):
        if any(r.traits != rs[0].traits for r in rs[1:]):
            raise ValueError('Incompatible matrix types')

        ixdtype = self.backend.ixdtype
        nblocks, nrow, *_, fpdtype = rs[0].traits
        ncola = rs[0].ioshape[-2]

        tplargs = dict(norm=norm, ncola=ncola, method=method)

        if method == 'resid':
            tplargs['dt_type'] = 'matrix' if dt_mat else 'scalar'

        # Render the reduction kernel template
        src = self.backend.lookup.get_template('reduction').render(**tplargs)

        # Array for the reduced data
        reduced = np.zeros(ncola, dtype=fpdtype)

        regs = list(rs) + [dt_mat] if dt_mat else rs

        # Argument types for reduction kernel
        if method == 'errest':
            argt = [ixdtype]*2 + [np.uintp]*4 + [fpdtype]*2
        elif method == 'resid' and dt_mat:
            argt = [ixdtype]*2 + [np.uintp]*4 + [fpdtype]
        else:
            argt = [ixdtype]*2 + [np.uintp]*3 + [fpdtype]

        # Build
        rkern = self._build_kernel('reduction', src, argt)
        rkern.set_args(nrow, nblocks, reduced.ctypes.data, *regs)

        # Runtime argument offset
        facoff = argt.index(fpdtype)

        class ReductionKernel(OpenMPKernel):
            @property
            def retval(self):
                return reduced

            def bind(self, *facs):
                self.kernel.set_args(*facs, start=facoff)

        return ReductionKernel(mats=regs, kernel=rkern)
