from math import prod

from pyfr.backends.base.generator import BaseKernelGenerator


class OpenMPKernelGenerator(BaseKernelGenerator):
    def render(self):
        kargdefn, kargassn = self._render_args('args')

        if self.ndim == 1:
            core = f'''
                for (int _xi = 0; _xi < BLK_SZ; _xi += SOA_SZ)
                {{
                    #pragma omp simd
                    for (int _xj = 0; _xj < SOA_SZ; _xj++)
                    {{
                        {self.body}
                    }}
                }}'''
            clean = f'''
                int _xi = 0;
                #pragma omp simd
                for (int _xj = 0; _xj < _nx % BLK_SZ; _xj++)
                {{
                    {self.body}
                }}'''
        else:
            core = f'''
                for (int _y = 0; _y < _ny; _y++)
                {{
                    for (int _xi = 0; _xi < BLK_SZ; _xi += SOA_SZ)
                    {{
                        #pragma omp simd
                        for (int _xj = 0; _xj < SOA_SZ; _xj++)
                        {{
                            {self.body}
                        }}
                    }}
                }}'''
            clean = f'''
                for (int _y = 0, _xi = 0; _y < _ny; _y++)
                {{
                    #pragma omp simd
                    for (int _xj = 0; _xj < _nx % BLK_SZ; _xj++)
                    {{
                        {self.body}
                    }}
                }}'''

        return f'''
            struct {self.name}_kargs {{ {kargdefn}; }};
            void {self.name}(int _ib, const struct {self.name}_kargs *args,
                             int _disp_mask)
            {{
                {kargassn};
                #define X_IDX (_xi + _xj)
                #define X_IDX_AOSOA(v, nv)\
                    ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
                #define BCAST_BLK(r, c, ld) ((c) % (ld) + ((c) / (ld))*(ld)*r)
                if (_nx - _ib*BLK_SZ >= BLK_SZ)
                {{
                    {core}
                }}
                else
                {{
                    {clean}
                }}
                #undef X_IDX
                #undef X_IDX_AOSOA
                #undef BCAST_BLK
            }}'''

    def ldim_size(self, name, factor=1):
        return f'{factor}*BLK_SZ' if factor > 1 else 'BLK_SZ'

    def needs_ldim(self, arg):
        return False

    def _displace_arg(self, arg):
        if arg.isview:
            return None
        elif self.ndim == 1:
            # Vector
            if arg.ncdim == 0 or arg.ismpi:
                return '_ib*BLK_SZ'
            # 2D broadcast vector
            elif arg.isbroadcast:
                return None
            # Stacked vector:
            else:
                return f'_ib*BLK_SZ*{prod(arg.cdims)}'
        else:
            # 2D broadcast vector or row broadcast matrix
            if arg.isbroadcast or arg.isbroadcastr:
                return None
            # Column broadcast matrix
            elif arg.isbroadcastc:
                return f'_ib*BLK_SZ*{prod(arg.cdims)}'
            # Matrix
            else:
                return f'_ib*BLK_SZ*{prod(arg.cdims)}*_ny'

    def _render_args(self, argn):
        # We first need the argument list; starting with the dimensions
        kargs = [('int ', d, None) for d in self._dims]

        # Now add any scalar arguments
        kargs.extend((sa.dtype, sa.name, None) for sa in self.scalargs)

        # Finally, add the vector arguments
        for i, va in enumerate(self.vectargs):
            disp = (i, da) if (da := self._displace_arg(va)) else None

            if va.intent == 'in':
                kargs.append((f'const {va.dtype}*', f'{va.name}_v', disp))
            else:
                kargs.append((f'{va.dtype}*', f'{va.name}_v', disp))

            # Views
            if va.isview:
                kargs.append(('const int*', f'{va.name}_vix',
                             (i, '_ib*BLK_SZ')))

                if va.ncdim == 2:
                    kargs.append(('const int*', f'{va.name}_vrstri',
                                 (i, '_ib*BLK_SZ')))

        # Argument definitions
        kargdefn = ';\n'.join(f'{dtype} {name}' for dtype, name, disp in kargs)

        # Argument assignments plus any required displacements
        kargassn = []
        for dtype, name, disp in kargs:
            assn = f'{dtype} {name} = {argn}->{name}'
            if disp:
                assn += f' + ((_disp_mask & {1 << disp[0]}) ? 0 : {disp[1]})'

            kargassn.append(assn)

        return kargdefn, ';\n'.join(kargassn)
