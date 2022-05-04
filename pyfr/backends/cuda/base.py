# -*- coding: utf-8 -*-

import re

from pyfr.backends.base import BaseBackend
from pyfr.mpiutil import get_local_rank


class CUDABackend(BaseBackend):
    name = 'cuda'
    blocks = False

    def __init__(self, cfg):
        super().__init__(cfg)

        from pyfr.backends.cuda.compiler import NVRTC
        from pyfr.backends.cuda.driver import CUDA, CUDAError

        # Load and wrap CUDA and NVRTC
        self.cuda = CUDA()
        self.nvrtc = NVRTC()

        # Get the desired CUDA device
        devid = cfg.get('backend-cuda', 'device-id', 'local-rank')
        if not re.match(r'(round-robin|local-rank|\d+)$', devid):
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
        else:
            self.cuda.set_device(int(devid))

        # Take the required alignment to be 128 bytes
        self.alignb = 128

        # Take the SoA size to be 32 elements
        self.soasz = 32
        self.csubsz = self.soasz

        # Get the MPI runtime type
        self.mpitype = cfg.get('backend-cuda', 'mpi-type', 'standard')
        if self.mpitype not in {'standard', 'cuda-aware'}:
            raise ValueError('Invalid CUDA backend MPI type')

        # Some CUDA devices share L1 cache and shared memory; on these
        # devices CUDA allows us to specify a preference between L1
        # cache and shared memory.  For the sake of CUBLAS (which
        # benefits greatly from more shared memory but fails to
        # declare its preference) we set the global default to
        # PREFER_SHARED.
        self.cuda.set_cache_pref(prefer_shared=True)

        from pyfr.backends.cuda import (blasext, cublas, gimmik, packing,
                                        provider, types)

        # Register our data types
        self.base_matrix_cls = types.CUDAMatrixBase
        self.const_matrix_cls = types.CUDAConstMatrix
        self.matrix_cls = types.CUDAMatrix
        self.matrix_slice_cls = types.CUDAMatrixSlice
        self.queue_cls = types.CUDAQueue
        self.view_cls = types.CUDAView
        self.xchg_matrix_cls = types.CUDAXchgMatrix
        self.xchg_view_cls = types.CUDAXchgView

        # Instantiate the base kernel providers
        kprovs = [provider.CUDAPointwiseKernelProvider,
                  blasext.CUDABlasExtKernels,
                  packing.CUDAPackingKernels,
                  gimmik.CUDAGiMMiKKernels,
                  cublas.CUDACUBLASKernels]
        self._providers = [k(self) for k in kprovs]

        # Pointwise kernels
        self.pointwise = self._providers[0]

    def _malloc_impl(self, nbytes):
        # Allocate
        data = self.cuda.mem_alloc(nbytes)

        # Zero
        self.cuda.memset(data, 0, nbytes)

        return data
