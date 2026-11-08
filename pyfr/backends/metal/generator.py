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
    _ptr_arg = 'device {c}{t}* {n}'
    _shared_sync = 'threadgroup_barrier(mem_flags::mem_threadgroup)'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Copy the packed scalar arguments out into named locals
        unpack = '\n'.join(f'const {t} {n} = _sa.{n};'
                           for p, _, t, n in self._kargs() if not p)
        self.preamble = f'{unpack}\n{self.preamble}'

    def _render_spec(self):
        kargs = self._kargs()

        # Pack the scalar arguments into a single constant buffer
        fields = ''.join(f'{t} {n}; ' for p, _, t, n in kargs if not p)
        kargs = ['constant _sargs_t& _sa',
                 *(self._karg(*a) for a in kargs if a[0])]

        # Finally, the attribute arguments
        kargs.append('uint2 _tpig [[thread_position_in_grid]]')
        if re.search(r'\b_tpitg\b', self.preamble):
            kargs.append('uint2 _tpitg [[thread_position_in_threadgroup]]')

        return 'struct _sargs_t {{ {0}}};\nkernel void {1}({2})'.format(
            fields, self.name, ', '.join(kargs)
        )
