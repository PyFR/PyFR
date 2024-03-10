from pyfr.backends.base.generator import BaseGPUKernelGenerator


class OpenCLKernelGenerator(BaseGPUKernelGenerator):
    _lid = ('get_local_id(0)', 'get_local_id(1)')
    _gid = 'get_global_id(0)'
    _shared_prfx = '__local'
    _shared_sync = 'work_group_barrier(CLK_GLOBAL_MEM_FENCE)'

    def _render_spec(self):
        g, c, r = '__global', 'const', 'restrict'

        # We first need the argument list; starting with the dimensions
        kargs = [f'ixdtype_t {d}' for d in self._dims]

        # Now add any scalar arguments
        kargs.extend(f'{sa.dtype} {sa.name}' for sa in self.scalargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            if va.intent == 'in':
                kargs.append(f'{g} {c} {va.dtype}* {r} {va.name}_v')
            else:
                kargs.append(f'{g} {va.dtype}* {r} {va.name}_v')

            # Views
            if va.isview:
                kargs.append(f'{g} {c} ixdtype_t* {r} {va.name}_vix')

                if va.ncdim == 2:
                    kargs.append(f'{g} {c} ixdtype_t* {r} {va.name}_vrstri')
            # Arrays
            elif self.needs_ldim(va):
                kargs.append(f'ixdtype_t ld{va.name}')

        # Determine the work group size for the kernel
        wgs = (*self.block1d, 1, 1) if self.ndim == 1 else (*self.block2d, 1)
        wgs = ', '.join(str(s) for s in wgs)
        kattrs = f'__kernel __attribute__((reqd_work_group_size({wgs})))'

        return '{0} void {1}({2})'.format(kattrs, self.name, ', '.join(kargs))
