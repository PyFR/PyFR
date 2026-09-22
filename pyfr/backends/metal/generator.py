from functools import cached_property
import re

from pyfr.backends.base.generator import BaseGPUKernelGenerator
from pyfr.dsl.codegen import CodeGenerator


class MetalCodeGenerator(CodeGenerator):
    region_markers = {
        'fp-precise': '_Pragma("clang fp reassociate(off) contract(off)")'
    }


class MetalKernelGenerator(BaseGPUKernelGenerator):
    codegen_cls = MetalCodeGenerator

    _lid = ('_tpitg.x', '_tpitg.y')
    _gid = '_tpig.x'
    _shared_prfx = 'threadgroup'
    _shared_sync = 'threadgroup_barrier(mem_flags::mem_threadgroup)'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Copy the packed scalar arguments out into named locals
        unpack = '\n'.join(f'const {t} {n} = _sa.{n};'
                           for t, n in self._kargs[0])
        self.preamble = f'{unpack}\n{self.preamble}'

    @cached_property
    def _kargs(self):
        # Split the arguments into packed scalars and buffer pointers
        sargs = [('ixdtype_t', d) for d in self._dims]
        sargs += [(sa.dtype, sa.name) for sa in self.scalargs]
        pargs = []

        for va in self.vectargs:
            if va.intent == 'in':
                pargs.append(f'device const {va.dtype}* {va.name}_v')
            else:
                pargs.append(f'device {va.dtype}* {va.name}_v')

            # Views
            if va.isview:
                pargs.append(f'device const ixdtype_t* {va.name}_vix')

                if self.ndim == 2 and not va.isbroadcastc:
                    sargs.append(('ixdtype_t', f'{va.name}_vrstri'))
                elif va.ncdim == 2 and va.cdims[0] > 1:
                    pargs.append(f'device const ixdtype_t* {va.name}_vrstri')
            # Arrays
            elif self.needs_ldim(va):
                sargs.append(('ixdtype_t', f'ld{va.name}'))

        return sargs, pargs

    def _render_spec(self):
        sargs, pargs = self._kargs

        # Pack the scalar arguments into a single constant buffer
        fields = ''.join(f'{t} {n}; ' for t, n in sargs)
        kargs = ['constant _sargs_t& _sa', *pargs]

        # Finally, the attribute arguments
        kargs.append('uint2 _tpig [[thread_position_in_grid]]')
        if re.search(r'\b_tpitg\b', self.preamble):
            kargs.append('uint2 _tpitg [[thread_position_in_threadgroup]]')

        return 'struct _sargs_t {{ {0}}};\nkernel void {1}({2})'.format(
            fields, self.name, ', '.join(kargs)
        )
