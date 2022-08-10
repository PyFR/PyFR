# -*- coding: utf-8 -*-

import re

from pyfr.backends.base import BaseBackend
from pyfr.mpiutil import get_local_rank


class HIPBackend(BaseBackend):
    name = 'hip'
    blocks = False

    def __init__(self, cfg):
        super().__init__(cfg)

        from pyfr.backends.hip.compiler import HIPRTC
        from pyfr.backends.hip.driver import HIP

        # Load and wrap HIP and HIPRTC
        self.hip = HIP()
        self.hiprtc = HIPRTC()

        # Get the desired HIP device
        devid = cfg.get('backend-hip', 'device-id', 'local-rank')
        if not re.match(r'(local-rank|\d+)$', devid):
            raise ValueError('Invalid device-id')

        # Handle the local-rank case
        if devid == 'local-rank':
            devid = str(get_local_rank())

        # Set the device
        self.hip.set_device(int(devid))

        # Get its properties
        self.props = self.hip.device_properties(int(devid))

        # Take the required alignment to be 128 bytes
        self.alignb = 128

        # Take the SoA size to be 32 elements
        self.soasz = 32
        self.csubsz = self.soasz

        # Get the MPI runtime type
        self.mpitype = cfg.get('backend-hip', 'mpi-type', 'standard')
        if self.mpitype not in {'standard', 'hip-aware'}:
            raise ValueError('Invalid HIP backend MPI type')

        from pyfr.backends.hip import (blasext, gimmik, packing, provider,
                                       rocblas, types)

        # Register our data types and meta kernels
        self.const_matrix_cls = types.HIPConstMatrix
        self.graph_cls = types.HIPGraph
        self.matrix_cls = types.HIPMatrix
        self.matrix_slice_cls = types.HIPMatrixSlice
        self.view_cls = types.HIPView
        self.xchg_matrix_cls = types.HIPXchgMatrix
        self.xchg_view_cls = types.HIPXchgView
        self.ordered_meta_kernel_cls = provider.HIPOrderedMetaKernel
        self.unordered_meta_kernel_cls = provider.HIPUnorderedMetaKernel

        # Instantiate the base kernel providers
        kprovs = [provider.HIPPointwiseKernelProvider,
                  blasext.HIPBlasExtKernels,
                  packing.HIPPackingKernels,
                  gimmik.HIPGiMMiKKernels,
                  rocblas.HIPRocBLASKernels]
        self._providers = [k(self) for k in kprovs]

        # Pointwise kernels
        self.pointwise = self._providers[0]

        # Create a stream to run kernels on
        self._stream = self.hip.create_stream()

    def run_kernels(self, kernels, wait=False):
        # Submit the kernels to the HIP stream
        for k in kernels:
            k.run(self._stream)

        if wait:
            self._stream.synchronize()

    def run_graph(self, graph, wait=False):
        graph.run(self._stream)

        if wait:
            self._stream.synchronize()

    def _malloc_impl(self, nbytes):
        # Allocate
        data = self.hip.mem_alloc(nbytes)

        # Zero
        self.hip.memset(data, 0, nbytes)

        return data
