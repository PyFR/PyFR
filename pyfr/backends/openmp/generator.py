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
               struct kargs {{ {kargdefn}; }};
               void {self.name}(const struct kargs *restrict args)
               {{
                   {kargassn};
                   #define X_IDX (_xi + _xj)
                   #define X_IDX_AOSOA(v, nv)\
                       ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
                   #define BLK_IDX (_ib*BLK_SZ)
                   #define BCAST_BLK(r, c, ld)\
                       ((c) % (ld) + ((c) / (ld))*(ld)*r)
                   #pragma omp parallel for {self.schedule}
                   for (int _ib = 0; _ib < _nx / BLK_SZ; _ib++)
                   {{
                       {core}
                   }}
                   int _ib = _nx / BLK_SZ;
                   {clean}
                   #undef X_IDX
                   #undef X_IDX_AOSOA
                   #undef BLK_IDX
                   #undef BCAST_BLK
               }}'''

    def ldim_size(self, name, factor=1):
        return f'{factor}*BLK_SZ' if factor > 1 else 'BLK_SZ'

    def needs_ldim(self, arg):
        return False

    def _render_args(self, argn):
        # We first need the argument list; starting with the dimensions
        kargs = [('int ', d) for d in self._dims]

        # Now add any scalar arguments
        kargs.extend((sa.dtype, sa.name) for sa in self.scalargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            if va.intent == 'in':
                kargs.append((f'const {va.dtype}*', f'{va.name}_v'))
            else:
                kargs.append((f'{va.dtype}*', f'{va.name}_v'))

            # Views
            if va.isview:
                kargs.append(('const int*', f'{va.name}_vix'))

                if va.ncdim == 2:
                    kargs.append(('const int*', f'{va.name}_vrstri'))

        # Argument definition and assignment operations
        kargdefn = ';\n'.join(f'{t} {n}' for t, n in kargs)
        kargassn = ';\n'.join(f'{t} {n} = {argn}->{n}' for t, n in kargs)

        return kargdefn, kargassn
