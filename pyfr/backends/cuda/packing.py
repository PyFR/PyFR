from pyfr.backends.base import NullKernel
from pyfr.backends.cuda.provider import (CUDAKernel, CUDAKernelProvider,
                                         get_grid_for_block)


class CUDAPackingKernels(CUDAKernelProvider):
    def pack(self, mv):
        cuda = self.backend.cuda

        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        src = self.backend.lookup.get_template('pack').render()

        # Build
        kern = self._build_kernel('pack_view', src, 'iiiPPPP')

        # Compute the grid and thread-block size
        block = (128, 1, 1)
        grid = get_grid_for_block(block, v.n)

        # Set the arguments
        params = kern.make_params(grid, block)
        params.set_args(v.n, v.nvrow, v.nvcol, v.basedata, v.mapping,
                        v.rstrides or 0, m)

        # If MPI is CUDA aware then we just need to pack the buffer
        if self.backend.mpitype == 'cuda-aware':
            class PackXchgViewKernel(CUDAKernel):
                def add_to_graph(self, graph, deps):
                    return graph.graph.add_kernel(params, deps)

                def run(self, stream):
                    kern.exec_async(stream, params)
        # Otherwise, we need to both pack the buffer and copy it back
        else:
            class PackXchgViewKernel(CUDAKernel):
                def add_to_graph(self, graph, deps):
                    gpack = graph.graph.add_kernel(params, deps)
                    return graph.graph.add_memcpy(
                        m.hdata, m.data, m.nbytes, [gpack]
                    )

                def run(self, stream):
                    kern.exec_async(stream, params)
                    cuda.memcpy(m.hdata, m.data, m.nbytes, stream)

        return PackXchgViewKernel(mats=[mv])

    def unpack(self, mv):
        cuda = self.backend.cuda

        if self.backend.mpitype == 'cuda-aware':
            return NullKernel()
        else:
            class UnpackXchgMatrixKernel(CUDAKernel):
                def add_to_graph(self, graph, deps):
                    return graph.graph.add_memcpy(mv.data, mv.hdata, mv.nbytes,
                                                  deps)

                def run(self, stream):
                    cuda.memcpy(mv.data, mv.hdata, mv.nbytes, stream)

            return UnpackXchgMatrixKernel(mats=[mv])
