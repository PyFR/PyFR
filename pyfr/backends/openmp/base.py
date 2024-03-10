from ctypes import c_int, c_void_p
from functools import cached_property
import re

import numpy as np

from pyfr.backends.base import BaseBackend
from pyfr.backends.openmp.compiler import OpenMPCompiler


class OpenMPBackend(BaseBackend):
    name = 'openmp'
    blocks = True

    def __init__(self, cfg):
        super().__init__(cfg)

        # Take the default alignment requirement to be 64-bytes
        self.alignb = cfg.getint('backend-openmp', 'alignb', 64)
        if self.alignb < 32 or (self.alignb & (self.alignb - 1)):
            raise ValueError('Alignment must be a power of 2 and >= 32')

        # Compute the SoA and AoSoA size
        self.soasz = self.alignb // np.dtype(self.fpdtype).itemsize
        self.csubsz = self.soasz*cfg.getint('backend-openmp', 'n-soa', 1)

        # OpenMP schedule
        sched = cfg.get('backend-openmp', 'schedule', 'static')
        if not re.match(r'(static|(dynamic|guided)(\s*,\s*\d+)?)$', sched):
            raise ValueError('Invalid OpenMP schedule')

        self.schedule = f'schedule({sched})'

        # C source compiler
        self.compiler = OpenMPCompiler(cfg)

        from pyfr.backends.openmp import (blasext, packing, provider, types,
                                          xsmm)

        # Register our data types and meta kernels
        self.const_matrix_cls = types.OpenMPConstMatrix
        self.graph_cls = types.OpenMPGraph
        self.matrix_cls = types.OpenMPMatrix
        self.matrix_slice_cls = types.OpenMPMatrixSlice
        self.view_cls = types.OpenMPView
        self.xchg_matrix_cls = types.OpenMPXchgMatrix
        self.xchg_view_cls = types.OpenMPXchgView
        self.ordered_meta_kernel_cls = provider.OpenMPOrderedMetaKernel
        self.unordered_meta_kernel_cls = provider.OpenMPUnorderedMetaKernel

        # Instantiate mandatory kernel provider classes
        kprovcls = [provider.OpenMPPointwiseKernelProvider,
                    blasext.OpenMPBlasExtKernels,
                    packing.OpenMPPackingKernels,
                    xsmm.OpenMPXSMMKernels]
        self._providers = [k(self) for k in kprovcls]

        # Pointwise kernels
        self.pointwise = self._providers[0]

    def run_kernels(self, kernels, wait=False):
        for k in kernels:
            k.run()

    def run_graph(self, graph, wait=False):
        graph.run()

    def wait(self):
        pass

    @cached_property
    def lookup(self):
        lookup = super().lookup
        lookup.dfltargs['schedule'] = self.schedule

        return lookup

    @cached_property
    def krunner(self):
        ksrc = self.lookup.get_template('run-kernels').render()
        klib = self.compiler.build(ksrc)
        return klib.function('run_kernels', None, [c_int, c_void_p])

    def _malloc_impl(self, nbytes):
        data = np.zeros(nbytes + self.alignb, dtype=np.uint8)
        offset = -data.ctypes.data % self.alignb

        return data[offset:nbytes + offset]
