from math import prod

from pyfr.backends.base.generator import BaseGPUKernelGenerator
from pyfr.dsl.codegen import CodeGenerator


class HIPCodeGenerator(CodeGenerator):
    region_markers = {
        'fp-precise': '_Pragma("clang fp reassociate(off) contract(off)")'
    }
    intrinsics = {'sync': '__syncthreads()'}


class HIPKernelGenerator(BaseGPUKernelGenerator):
    block1d = (64, 1, 1)
    block2d = (64, 4, 1)
    codegen_cls = HIPCodeGenerator

    _lid = ('threadIdx.x', 'threadIdx.y')
    _gid = 'PYFR_GLOBAL_ID_X'
    _grp = 'ixdtype_t(PYFR_BLOCK_ID)'
    _shared_prfx = '__shared__'
    _function_qual = '__device__ '
    _ptr_arg = '{c}{t}* __restrict__ {n}'

    def _render_spec(self):
        kargs = [self._karg(*a) for a in self._kargs()]

        # Determine the launch bounds for the kernel
        nthrds = prod(self.block)
        kattrs = f'__global__ __launch_bounds__({nthrds})'

        return '{0} void {1}({2})'.format(kattrs, self.name, ', '.join(kargs))
