import numpy as np

from pyfr.backends.base import NullKernel
from pyfr.backends.cuda.provider import (CUDAKernel, CUDAKernelProvider,
                                         get_grid_for_block)


class CUDAPackingKernels(CUDAKernelProvider):
    def _extract_args(self, xmv):
        if isinstance(xmv, self.backend.xchg_view_cls):
            return xmv.xchgmat, xmv.view
        else:
            return xmv, None

    def _packing_kern(self, type, v, xm):
        # Render the kernel template
        src = self.backend.lookup.get_template('packing').render()

        # Build
        kern = self._build_kernel(f'{type}_view', src,
                                  [self.backend.ixdtype]*3 + [np.uintp]*4)

        # Compute the grid and thread-block size
        block = (128, 1, 1)
        grid = get_grid_for_block(block, v.n)

        # Set the arguments
        params = kern.make_params(grid, block)
        params.set_args(v.n, v.nvrow, v.nvcol, v.basedata, v.mapping,
                        v.rstrides or 0, xm)

        return kern, params

    def pack(self, xmv):
        cuda = self.backend.cuda
        xm, v = self._extract_args(xmv)

        # If we have a view then obtain a view-packing kernel
        if v:
            kern, params = self._packing_kern('pack', v, xm)

        # When copy elision is requested just pack the buffer
        if xm.elide_copy:
            # If we have been passed a view then pack it
            if v:
                class PackKernel(CUDAKernel):
                    def add_to_graph(self, graph, deps):
                        return graph.graph.add_kernel(params, deps)

                    def run(self, stream):
                        kern.exec_async(stream, params)
            # Otherwise, there is nothing to do
            else:
                return NullKernel()
        # Otherwise, we need to potentially pack the buffer and copy it back
        else:
            class PackKernel(CUDAKernel):
                def add_to_graph(self, graph, deps):
                    if v:
                        deps = [graph.graph.add_kernel(params, deps)]

                    return graph.graph.add_memcpy(xm.hdata, xm.data, xm.nbytes,
                                                  deps)

                def run(self, stream):
                    if v:
                        kern.exec_async(stream, params)

                    cuda.memcpy(xm.hdata, xm.data, xm.nbytes, stream)

        return PackKernel(mats=[xmv])

    def unpack(self, xmv):
        cuda = self.backend.cuda
        xm, v = self._extract_args(xmv)

        # If we have a view then obtain a view-unpacking kernel
        if v:
            kern, params = self._packing_kern('unpack', v, xm)

        # When copy elision is requested there is no need to copy the buffer
        if xm.elide_copy:
            # If we have been passed a view then unpack it
            if v:
                class UnpackKernel(CUDAKernel):
                    def add_to_graph(self, graph, deps):
                        return graph.graph.add_kernel(params, deps)

                    def run(self, stream):
                        kern.exec_async(stream, params)
            # Otherwise, there is nothing to do
            else:
                return NullKernel()
        # Otherwise, we need to copy the host buffer and potentially unpack
        else:
            class UnpackKernel(CUDAKernel):
                def add_to_graph(self, graph, deps):
                    gcopy = graph.graph.add_memcpy(xm.data, xm.hdata,
                                                   xm.nbytes, deps)

                    if v:
                        return graph.graph.add_kernel(params, [gcopy])
                    else:
                        return gcopy

                def run(self, stream):
                    cuda.memcpy(xm.data, xm.hdata, xm.nbytes, stream)

                    if v:
                        kern.exec_async(stream, params)

        return UnpackKernel(mats=[xmv])
