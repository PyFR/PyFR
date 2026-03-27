import numpy as np

from pyfr.backends.base.blasext import BaseBlasExtKernels
from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider


class OpenCLBlasExtKernels(BaseBlasExtKernels, OpenCLKernelProvider):
    pvar_idx = 'k'

    def batched_inv(self, m):
        class BatchedInvKernel(OpenCLKernel):
            def run(self, queue, wait_for=None, ret_evt=False):
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
        kern.set_dims((ncolb + (-ncolb % 128), nrow), (128, 1))
        kern.set_args(ncolb, ldim, *arr)

        class AxnpbyKernel(OpenCLKernel):
            def bind(self, *consts):
                kern.set_args(*consts, start=2 + nv)

            def run(self, queue, wait_for=None, ret_evt=False):
                return kern.exec_async(queue, wait_for, ret_evt)

        return AxnpbyKernel(mats=arr)

    def copy(self, dst, src):
        cl = self.backend.cl

        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(OpenCLKernel):
            def run(self, queue, wait_for=None, ret_evt=False):
                return cl.memcpy(queue, dst, src, dst.nbytes, blocking=False,
                                 wait_for=wait_for, ret_evt=ret_evt)

        return CopyKernel(mats=[dst, src])

    def zero(self, m):
        cl = self.backend.cl

        class ZeroKernel(OpenCLKernel):
            def run(self, queue, wait_for=None, ret_evt=False):
                return cl.zero(m, m.nbytes, queue,
                               wait_for=wait_for, ret_evt=ret_evt)

        return ZeroKernel(mats=[m])

    def _reduction(self, fvvar, vvars, svars, tplargs):
        cl = self.backend.cl
        ixdtype = self.backend.ixdtype
        nrow, _, ldim, fpdtype = fvvar.traits[1:]
        ncola, ncolb = fvvar.ioshape[-2:]
        nexprs = tplargs['nexprs']

        # Reduction workgroup dimensions
        ls = (128, 1)
        gs = (ncolb - ncolb % -ls[0], ncola)

        # Empty result buffer on host with (nexprs, ncola, ngroups)
        reduced_host = np.empty((nexprs, ncola, gs[0] // ls[0]), fpdtype)

        # Corresponding device memory allocation
        reduced_dev = cl.mem_alloc(reduced_host.nbytes)

        # Add backend-specific template arguments
        tplargs['ncola'] = ncola

        # Get the kernel template
        src = self.backend.lookup.get_template('reduction').render(**tplargs)

        # Argument types for the reduction kernel
        argt = [ixdtype]*3 + [np.uintp]*(1 + len(vvars)) + [fpdtype]*len(svars)

        # Build the reduction kernel
        rkern = self._build_kernel('reduction', src, argt)
        rkern.set_dims(gs, ls)
        rkern.set_args(nrow, ncolb, ldim, reduced_dev, *vvars.values())

        # Runtime argument offset
        coff = 4 + len(vvars)

        # Reduction type
        reducer = np.max if tplargs['rop'] == 'max' else np.sum

        class ReductionKernel(OpenCLKernel):
            @property
            def retval(self):
                return reducer(reduced_host, axis=(1, 2))

            if svars:
                def bind(self, *consts):
                    rkern.set_args(*consts, start=coff)

            def run(self, queue, wait_for=None, ret_evt=False):
                revt = rkern.exec_async(queue, wait_for, True)
                return cl.memcpy(queue, reduced_host, reduced_dev,
                                 reduced_dev.nbytes, False, [revt], ret_evt)

        return ReductionKernel(mats=vvars.values())
