# -*- coding: utf-8 -*-

from pyfr.backends.base.generator import BaseKernelGenerator


class OpenMPKernelGenerator(BaseKernelGenerator):
    def render(self):
        if self.ndim == 1:
            core = f'''
                   for (int _xi = 0; _xi < SZ; _xi += SOA_SZ)
                   {{
                       #pragma omp simd
                       for (int _xj = 0; _xj < SOA_SZ; _xj++)
                       {{
                           {self.body}
                       }}
                   }}'''
            clean = f'''
                    for (int _xi = 0; _xi < (rem/SOA_SZ)*SOA_SZ; _xi += SOA_SZ)
                    {{
                        #pragma omp simd
                        for (int _xj = 0; _xj < SOA_SZ; _xj++)
                        {{
                            {self.body}
                        }}
                    }}
                    for (int _xi = (rem/SOA_SZ)*SOA_SZ, _xj = 0; _xj < rem%SOA_SZ; _xj++)
                    {{
                        {self.body}
                    }}'''
        else:
            core = f'''
                   for (int _y = 0; _y < _ny; _y++)
                   {{
                       for (int _xi = 0; _xi < SZ; _xi += SOA_SZ)
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
                        for (int _xi = 0; _xi < (rem/SOA_SZ)*SOA_SZ; _xi += SOA_SZ)
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
                        for (int _xi = (rem/SOA_SZ)*SOA_SZ, _xj = 0; _xj < rem%SOA_SZ; _xj++)
                        {{
                            {self.body}
                        }}
                    }}'''

        return f'''{self._render_spec()}
               {{
                   int nci = _nx / SZ;
                   int rem = _nx % SZ;
                   #define X_IDX (_xi + _xj)
                   #define X_IDX_AOSOA(v, nv)\
                       ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
                   #define BLK_IDX ib*SZ
                   #pragma omp parallel for
                   for (int ib = 0; ib < nci; ib++)
                   {{
                       {core}
                   }}
                   int ib = nci;
                   {clean}
                   #undef X_IDX
                   #undef X_IDX_AOSOA
               }}'''

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

                if self.needs_ldim(va):
                    kargs.append(f'int ld{va.name}')

        return f'void {self.name}({", ".join(kargs)})'
