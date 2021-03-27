# -*- coding: utf-8 -*-

from pyfr.backends.base.generator import BaseKernelGenerator


class OpenMPKernelGenerator(BaseKernelGenerator):
    def render(self):
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
                    for (int _xi = 0; _xi < _remi; _xi += SOA_SZ)
                    {{
                        #pragma omp simd
                        for (int _xj = 0; _xj < SOA_SZ; _xj++)
                        {{
                            {self.body}
                        }}
                    }}
                    for (int _xi = _remi, _xj = 0; _xj < _remj; _xj++)
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
                    for (int _y = 0; _y < _ny; _y++)
                    {{
                        for (int _xi = 0; _xi < _remi; _xi += SOA_SZ)
                        {{
                            #pragma omp simd
                            for (int _xj = 0; _xj < SOA_SZ; _xj++)
                            {{
                                {self.body}
                            }}
                        }}
                    }}
                    for (int _y = 0; _y < _ny; _y++)
                    {{
                        for (int _xi = _remi, _xj = 0; _xj < _remj; _xj++)
                        {{
                            {self.body}
                        }}
                    }}'''

        return f'''{self._render_spec()}
               {{
                   int nci = _nx / BLK_SZ;
                   int _remi = ((_nx % BLK_SZ) / SOA_SZ)*SOA_SZ;
                   int _remj = (_nx % BLK_SZ) % SOA_SZ;
                   #define X_IDX (_xi + _xj)
                   #define X_IDX_AOSOA(v, nv)\
                       ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
                   #define BLK_IDX ib*BLK_SZ
                   #define BCAST_BLK(i, ld)\
                       ((i) % (ld) + ((i) / (ld))*(ld)*_ny)
                   #pragma omp parallel for
                   for (int ib = 0; ib < nci; ib++)
                   {{
                       {core}
                   }}
                   int ib = nci;
                   {clean}
                   #undef X_IDX
                   #undef X_IDX_AOSOA
                   #undef BLK_IDX
                   #undef BCAST_BLK
               }}'''

    def ldim_size(self, name, *factor):
        return '*'.join(['BLK_SZ'] + [str(f) for f in factor])

    def needs_ldim(self, arg):
        return False

    def _render_spec(self):
        # We first need the argument list; starting with the dimensions
        kargs = ['int ' + d for d in self._dims]

        # Now add any scalar arguments
        kargs.extend(f'{sa.dtype} {sa.name}' for sa in self.scalargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            # Views
            if va.isview:
                kargs.append(f'{va.dtype}* __restrict__ {va.name}_v')
                kargs.append(f'const int* __restrict__ {va.name}_vix')

                if va.ncdim == 2:
                    kargs.append(f'const int* __restrict__ {va.name}_vrstri')
            # Arrays
            else:
                # Intent in arguments should be marked constant
                const = 'const' if va.intent == 'in' else ''

                kargs.append(f'{const} {va.dtype}* __restrict__ {va.name}_v'
                             .strip())

        return 'void {0}({1})'.format(self.name, ', '.join(kargs))
