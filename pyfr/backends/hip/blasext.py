import numpy as np

from pyfr.backends.base.blasext import BaseBlasExtKernels
from pyfr.backends.hip.provider import HIPKernel, HIPKernelProvider


class HIPBlasExtKernels(BaseBlasExtKernels, HIPKernelProvider):
    pvar_idx = 'VARIDX'

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

    def zero(self, m):
        hip = self.backend.hip

        class ZeroKernel(HIPKernel):
            def add_to_graph(self, graph, deps):
                return graph.graph.add_memset(m, 0, m.nbytes, deps)

            def run(self, stream):
                hip.memset(m, 0, m.nbytes, stream)

        return ZeroKernel(mats=[m])

    def _reduction(self, fvvar, vvars, svars, tplargs):
        hip = self.backend.hip
        ixdtype = self.backend.ixdtype
        nrow, _, ldim, fpdtype = fvvar.traits[1:]
        ncola, ncolb = fvvar.ioshape[-2:]
        nexprs = tplargs['nexprs']

        # Reduction block dimensions
        block = (256, 1, 1)
        nblocks = min(1024, (ncolb + block[0] - 1) // block[0])
        grid = (nblocks, ncola, 1)

        # Result buffer on device
        reduced_dev = hip.mem_alloc(nexprs*ncola*fvvar.itemsize)

        # Result buffer on host
        reduced_host = hip.pagelocked_empty((nexprs, ncola), fpdtype)

        # Initialisation buffer (0 for sum, -fpdtype_max for max)
        init_host = hip.pagelocked_empty((nexprs, ncola), fpdtype)
        init_host.fill(tplargs['init_val'])

        # Add backend-specific template arguments
        tplargs['blocksz'] = block[0]
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

        class ReductionKernel(HIPKernel):
            @property
            def retval(self):
                return reducer(reduced_host, axis=1)

            if svars:
                def bind(self, *consts):
                    params.set_args(*consts, start=coff)

            def run(self, stream):
                hip.memcpy(reduced_dev, init_host, reduced_dev.nbytes, stream)
                rkern.exec_async(stream, params)
                hip.memcpy(reduced_host, reduced_dev, reduced_dev.nbytes,
                           stream)

        return ReductionKernel(mats=vvars.values())
