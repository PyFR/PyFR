import re

from pyfr.backends.base.generator import BaseGPUKernelGenerator
from pyfr.dsl.codegen import CodeGenerator


class MetalCodeGenerator(CodeGenerator):
    region_markers = {
        'fp-precise': '_Pragma("clang fp reassociate(off) contract(off)")',
        'unroll': '#pragma unroll {0}', 'simd': ''
    }
    intrinsics = {'sync': 'threadgroup_barrier(mem_flags::mem_threadgroup)',
                  'sync_gmem': 'PYFR_SYNC_GMEM_THREADS()'}


class MetalKernelGenerator(BaseGPUKernelGenerator):
    block1d = (128, 1, 1)
    block2d = (32, 8, 1)
    codegen_cls = MetalCodeGenerator

    _lid = ('_tpitg.x', '_tpitg.y')
    _gid = '_tpig.x'
    _grp = '_tgpig.x'
    _shared_prfx = 'threadgroup'
    _ptr_arg = 'device {c}{t}* {n}'

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

        # Finally, the attribute arguments referenced by the kernel
        src = (f'{self._lane_decls()} {self._prologue} {self._item} '
               f'{self.preamble} {self.body} {self.epilogue}')
        kargs.append('uint2 _tpig [[thread_position_in_grid]]')
        if re.search(r'\b_tpitg\b', src):
            kargs.append('uint2 _tpitg [[thread_position_in_threadgroup]]')
        if re.search(r'\b_tgpig\b', src):
            kargs.append('uint2 _tgpig [[threadgroup_position_in_grid]]')

        return 'struct _sargs_t {{ {0}}};\nkernel void {1}({2})'.format(
            fields, self.name, ', '.join(kargs)
        )
