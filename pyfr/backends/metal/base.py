import numpy as np

from pyfr.backends.base import BaseBackend
from pyfr.backends.metal.util import call_


class MetalBackend(BaseBackend):
    name = 'metal'
    blocks = False

    def __init__(self, cfg):
        super().__init__(cfg)

        from Metal import MTLCreateSystemDefaultDevice

        # Get the default device
        self.dev = MTLCreateSystemDefaultDevice()

        # Metal does not support double precision arithmetic
        if self.fpdtype == np.float64:
            raise ValueError('Device does not support double precision')

        # Take the alignment requirement to be 128 bytes
        self.alignb = 128

        # Take the SoA size to be 32
        self.soasz = 32
        self.csubsz = self.soasz

        from pyfr.backends.metal import (blasext, gimmik, mps, packing,
                                         provider, types)

        # Register our data types and meta kernels
        self.const_matrix_cls = types.MetalConstMatrix
        self.graph_cls = types.MetalGraph
        self.matrix_cls = types.MetalMatrix
        self.matrix_slice_cls = types.MetalMatrixSlice
        self.view_cls = types.MetalView
        self.xchg_matrix_cls = types.MetalXchgMatrix
        self.xchg_view_cls = types.MetalXchgView
        self.ordered_meta_kernel_cls = provider.MetalOrderedMetaKernel
        self.unordered_meta_kernel_cls = provider.MetalUnorderedMetaKernel

        # Instantiate the base kernel providers
        kprovs = [provider.MetalPointwiseKernelProvider,
                  blasext.MetalBlasExtKernels,
                  packing.MetalPackingKernels,
                  gimmik.MetalGiMMiKKernels,
                  mps.MetalMPSKernels]
        self._providers = [k(self) for k in kprovs]

        # Pointwise kernels
        self.pointwise = self._providers[0]

        # Create a command queue
        self.queue = self.dev.newCommandQueue()

        # Track the last command buffer to the queue
        self.last_cbuf = None

    def run_kernels(self, kernels, wait=False):
        cbuf = self.queue.commandBuffer()

        for k in kernels:
            k.run(cbuf)

        cbuf.commit()

        if wait:
            cbuf.waitUntilCompleted()
            self.last_cbuf = None
        else:
            self.last_cbuf = cbuf

    def run_graph(self, graph, wait=False):
        cbuf = graph.run(self.queue)

        if wait:
            cbuf.waitUntilCompleted()
            self.last_cbuf = None
        else:
            self.last_cbuf = cbuf

    def wait(self):
        if self.last_cbuf:
            self.last_cbuf.waitUntilCompleted()
            self.last_cbuf = None

    def _malloc_impl(self, nbytes):
        from Metal import MTLResourceStorageModeManaged

        # Allocate the device buffer
        return call_(self.dev, 'newBufferWith', length=nbytes,
                     options=MTLResourceStorageModeManaged)
