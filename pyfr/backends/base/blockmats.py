# -*- coding: utf-8 -*-

from pyfr.backends.base.backend import traits
from pyfr.backends.base.kernels import ComputeMetaKernel


class BlockDiagMatrixKernels(object):
    def __init__(self, backend, cfg):
        self.backend = backend

    @traits(a={'blockdiag'})
    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        kerns = []
        for (ri, rj, ci, cj), m in zip(a.ranges, a.blocks):
            aslice = self.backend.auto_matrix(m)
            bslice = b.rslice(ci, cj)
            oslice = out.rslice(ri, rj)

            k = self.backend.kernel('mul', aslice, bslice, oslice, alpha, beta)
            kerns.append(k)

        return ComputeMetaKernel(kerns)
