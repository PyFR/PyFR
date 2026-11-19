from math import prod

from pyfr.backends.base.generator import BaseGPUKernelGenerator
from pyfr.dsl.codegen import CodeGenerator


class CUDACodeGenerator(CodeGenerator):
    intrinsics = {'sync': '__syncthreads()'}


class CUDAKernelGenerator(BaseGPUKernelGenerator):
    block1d = (64, 1, 1)
    block2d = (32, 8, 1)
    codegen_cls = CUDACodeGenerator

    _lid = ('threadIdx.x', 'threadIdx.y')
    _gid = 'ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x'
    _grp = 'ixdtype_t(blockIdx.x)'
    _shared_prfx = '__shared__'
    _function_qual = '__device__ '
    _ptr_arg = '{c}{t}* __restrict__ {n}'

    def _render_spec(self):
        kargs = [self._karg(*a) for a in self._kargs()]

        # Determine the launch bounds for the kernel
        nthrds = prod(self.block)
        kattrs = f'__global__ __launch_bounds__({nthrds})'

        return '{0} void {1}({2})'.format(kattrs, self.name, ', '.join(kargs))
