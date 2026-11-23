from math import prod
import re

from pyfr.backends.base.generator import BaseGPUKernelGenerator
from pyfr.dsl.codegen import CodeGenerator


class CUDACodeGenerator(CodeGenerator):
    region_markers = {'unroll': '#pragma unroll {0}', 'simd': ''}
    intrinsics = {'sync': '__syncthreads()',
                  'sync_gmem': 'PYFR_SYNC_GMEM_THREADS()'}


class CUDAKernelGenerator(BaseGPUKernelGenerator):
    block1d = (64, 1, 1)
    block2d = (32, 8, 1)
    codegen_cls = CUDACodeGenerator
    bytes_inflight = 128

    _lid = ('threadIdx.x', 'threadIdx.y')
    _gid = 'ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x'
    _grp = 'ixdtype_t(blockIdx.x)'
    _shared_prfx = '__shared__'
    _function_qual = '__device__ '
    _ptr_arg = '{c}{t}* __restrict__ {n}'

    @property
    def meta(self):
        # Bytes of block storage for the launch to provide
        return super().meta | {'storage': self._storage()[1]}

    def _storage(self):
        model, refs, nbytes = self.lanemodel, [], 0

        for d in model.decls:
            ext = model.extents(d)
            dims = [int(n) for n in re.findall(r'\[(\d+)\]', ext)]

            # Refer to the declaration at the next free bytes of the storage
            ptr = f'reinterpret_cast<{d.dtype} (*){ext}>(_bstore + {nbytes})'
            refs.append(f'auto &{d.name} = *{ptr};')

            # Round the bytes it takes up to the alignment of the storage
            nbytes += -(-prod(dims)*self.npdtype(d.dtype).itemsize // 16)*16

        return refs, nbytes

    def _lane_decls(self):
        refs = self._storage()[0]

        # Declare the storage of the block, which the launch sizes
        store = 'extern __shared__ __align__(16) char _bstore[];'

        return '\n'.join([store, *refs]) if refs else ''

    def _render_spec(self):
        kargs = [self._karg(*a) for a in self._kargs()]

        # Determine the launch bounds for the kernel
        kattrs = f'__global__ {self._launch_bounds}'

        return '{0} void {1}({2})'.format(kattrs, self.name, ', '.join(kargs))
