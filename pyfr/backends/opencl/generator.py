from pyfr.backends.base.generator import BaseGPUKernelGenerator
from pyfr.dsl.codegen import CodeGenerator


class OpenCLCodeGenerator(CodeGenerator):
    region_markers = {
        'fp-precise': ('#ifdef __clang__\n'
                       '_Pragma("clang fp reassociate(off) contract(off)")\n'
                       '#endif'),
        'unroll': '__attribute__((opencl_unroll_hint({0})))', 'simd': ''
    }
    intrinsics = {'sync': 'work_group_barrier(CLK_LOCAL_MEM_FENCE)',
                  'sync_gmem': 'PYFR_SYNC_GMEM_THREADS()'}


class OpenCLKernelGenerator(BaseGPUKernelGenerator):
    block1d = (64, 1, 1)
    block2d = (64, 4, 1)
    codegen_cls = OpenCLCodeGenerator

    _lid = ('get_local_id(0)', 'get_local_id(1)')
    _gid = 'get_global_id(0)'
    _grp = 'get_group_id(0)'
    _shared_prfx = '__local'
    _ptr_arg = '__global {c}{t}* restrict {n}'

    def _render_spec(self):
        kargs = [self._karg(*a) for a in self._kargs()]

        # Determine the work group size for the kernel
        wgs = ', '.join(str(s) for s in self.block)
        kattrs = f'__kernel __attribute__((reqd_work_group_size({wgs})))'

        return '{0} void {1}({2})'.format(kattrs, self.name, ', '.join(kargs))
