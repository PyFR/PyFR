import re

from pyfr.backends.base import BaseBackend
from pyfr.mpiutil import get_local_rank


class CUDABackend(BaseBackend):
    name = 'cuda'
    blocks = False

    def __init__(self, cfg):
        super().__init__(cfg)

        from pyfr.backends.cuda.compiler import CUDACompiler
        from pyfr.backends.cuda.driver import CUDA, CUDAError

        # Load and wrap CUDA
        self.cuda = CUDA()

        # Get the desired CUDA device
        devid = cfg.get('backend-cuda', 'device-id', 'local-rank')

        uuid = '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        if not re.match(rf'(round-robin|local-rank|\d+|{uuid})$', devid):
            raise ValueError('Invalid device-id')

        # For round-robin try each device until we find one that works
        if devid == 'round-robin':
            for i in range(self.cuda.device_count()):
                try:
                    self.cuda.set_device(i)
                    break
                except CUDAError:
                    pass
            else:
                raise RuntimeError('Unable to create a CUDA context')
        elif devid == 'local-rank':
            self.cuda.set_device(get_local_rank())
        elif '-' in devid:
            for i in range(self.cuda.device_count()):
                if str(self.cuda.device_uuid(i)) == devid:
                    self.cuda.set_device(i)
                    break
            else:
                raise RuntimeError(f'Unable to find CUDA device {devid}')
        else:
            self.cuda.set_device(int(devid))

        # CUDA Compiler
        self.compiler = CUDACompiler(self.cuda)

        # Take the required alignment to be 128 bytes
        self.alignb = 128

        # Take the SoA size to be 32 elements
        self.soasz = 32
        self.csubsz = self.soasz

        # Get the MPI runtime type
        self.mpitype = cfg.get('backend-cuda', 'mpi-type', 'standard')
        if self.mpitype not in {'standard', 'cuda-aware'}:
            raise ValueError('Invalid CUDA backend MPI type')

        from pyfr.backends.cuda import (blasext, cublaslt, gimmik, packing,
                                        provider, types)

        # Register our data types and meta kernels
        self.const_matrix_cls = types.CUDAConstMatrix
        self.graph_cls = types.CUDAGraph
        self.matrix_cls = types.CUDAMatrix
        self.matrix_slice_cls = types.CUDAMatrixSlice
        self.view_cls = types.CUDAView
        self.xchg_matrix_cls = types.CUDAXchgMatrix
        self.xchg_view_cls = types.CUDAXchgView
        self.ordered_meta_kernel_cls = provider.CUDAOrderedMetaKernel
        self.unordered_meta_kernel_cls = provider.CUDAUnorderedMetaKernel

        # Instantiate the base kernel providers
        kprovs = [provider.CUDAPointwiseKernelProvider,
                  blasext.CUDABlasExtKernels,
                  packing.CUDAPackingKernels,
                  gimmik.CUDAGiMMiKKernels,
                  cublaslt.CUDACUBLASLtKernels]
        self._providers = [k(self) for k in kprovs]

        # Pointwise kernels
        self.pointwise = self._providers[0]

        # Create a stream to run kernels on
        self._stream = self.cuda.create_stream()

    def run_kernels(self, kernels, wait=False):
        # Submit the kernels to the CUDA stream
        for k in kernels:
            k.run(self._stream)

        if wait:
            self._stream.synchronize()

    def run_graph(self, graph, wait=False):
        graph.run(self._stream)

        if wait:
            self._stream.synchronize()

    def wait(self):
        self._stream.synchronize()

    def _malloc_impl(self, nbytes):
        # Allocate
        data = self.cuda.mem_alloc(nbytes)

        # Zero
        self.cuda.memset(data, 0, nbytes)

        return data
