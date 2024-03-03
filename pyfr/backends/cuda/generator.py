from math import prod

from pyfr.backends.base.generator import BaseGPUKernelGenerator


class CUDAKernelGenerator(BaseGPUKernelGenerator):
    _lid = ('threadIdx.x', 'threadIdx.y')
    _gid = 'ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x'
    _shared_prfx = '__shared__'
    _shared_sync = '__syncthreads()'

    def _render_spec(self):
        res = '__restrict__'

        # We first need the argument list; starting with the dimensions
        kargs = [f'ixdtype_t {d}' for d in self._dims]

        # Now add any scalar arguments
        kargs.extend(f'{sa.dtype} {sa.name}' for sa in self.scalargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            if va.intent == 'in':
                kargs.append(f'const {va.dtype}* {res} {va.name}_v')
            else:
                kargs.append(f'{va.dtype}* {res} {va.name}_v')

            # Views
            if va.isview:
                kargs.append(f'const ixdtype_t* {res} {va.name}_vix')

                if va.ncdim == 2:
                    kargs.append(f'const ixdtype_t* {res} {va.name}_vrstri')
            # Arrays
            elif self.needs_ldim(va):
                kargs.append(f'ixdtype_t ld{va.name}')

        # Determine the launch bounds for the kernel
        nthrds = prod(self.block1d if self.ndim == 1 else self.block2d)
        kattrs = f'__global__ __launch_bounds__({nthrds})'

        return '{0} void {1}({2})'.format(kattrs, self.name, ', '.join(kargs))
