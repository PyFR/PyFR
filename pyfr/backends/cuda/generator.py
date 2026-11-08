from math import prod

from pyfr.backends.base.generator import BaseGPUKernelGenerator


class CUDAKernelGenerator(BaseGPUKernelGenerator):
    _lid = ('threadIdx.x', 'threadIdx.y')
    _gid = 'ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x'
    _shared_prfx = '__shared__'
    _shared_sync = '__syncthreads()'
    _function_qual = '__device__ '
    _ptr_arg = '{c}{t}* __restrict__ {n}'

    def _render_spec(self):
        kargs = [self._karg(*a) for a in self._kargs()]

        # Determine the launch bounds for the kernel
        nthrds = prod(self.block1d if self.ndim == 1 else self.block2d)
        kattrs = f'__global__ __launch_bounds__({nthrds})'

        return '{0} void {1}({2})'.format(kattrs, self.name, ', '.join(kargs))
