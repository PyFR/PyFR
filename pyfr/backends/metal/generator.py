import re

from pyfr.backends.base.generator import BaseGPUKernelGenerator


class MetalKernelGenerator(BaseGPUKernelGenerator):
    _lid = ('_tpitg.x', '_tpitg.y')
    _gid = '_tpig.x'
    _shared_prfx = 'threadgroup'
    _shared_sync = 'threadgroup_barrier(mem_flags::mem_threadgroup)'

    def _render_spec(self):
        # We first need the argument list; starting with the dimensions
        kargs = [f'constant int& {d}' for d in self._dims]

        # Now add any scalar arguments
        kargs.extend(f'constant {sa.dtype}& {sa.name}' for sa in self.scalargs)

        # Then vector arguments
        for va in self.vectargs:
            if va.intent == 'in':
                kargs.append(f'device const {va.dtype}* {va.name}_v')
            else:
                kargs.append(f'device {va.dtype}* {va.name}_v')

            # Views
            if va.isview:
                kargs.append(f'device const int* {va.name}_vix')

                if va.ncdim == 2:
                    kargs.append(f'device const int* {va.name}_vrstri')
            # Arrays
            elif self.needs_ldim(va):
                kargs.append(f'device const int& ld{va.name}')

        # Finally, the attribute arguments
        kargs.append('uint2 _tpig [[thread_position_in_grid]]')
        if re.search(r'\b_tpitg\b', self.preamble):
            kargs.append('uint2 _tpitg [[thread_position_in_threadgroup]]')

        return 'kernel void {0}({1})'.format(self.name, ', '.join(kargs))
