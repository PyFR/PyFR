# -*- coding: utf-8 -*-

from pyfr.backends.base.generator import BaseKernelGenerator


class OpenMPKernelGenerator(BaseKernelGenerator):
    def render(self):
        if self.ndim == 1:
            core = '''
                   for (int _xi = 0; _xi < SZ; _xi += SOA_SZ)
                   {{
                       #pragma omp simd
                       for (int _xj = 0; _xj < SOA_SZ; _xj++)
                       {{
                           {body}
                       }}
                   }}'''.format(body=self.body)
            clean = '''
                    for (int _xi = 0; _xi < rem; _xi += SOA_SZ)
                    {{
                        #pragma omp simd
                        for (int _xj = 0; _xj < SOA_SZ; _xj++)
                        {{
                            {body}
                        }}
                    }}
                    for (int _xi = (rem/SOA_SZ)*SOA_SZ, _xj = 0; _xj < rem%SOA_SZ; _xj++)
                    {{
                        {body}
                    }}'''.format(body=self.body)
        else:
            core = '''
                   for (int _xi = 0; _xi < SZ; _xi += SOA_SZ)
                   {{
                       for (int _y = 0; _y < _ny; _y++)
                       {{
                           #pragma omp simd
                           for (int _xj = 0; _xj < SOA_SZ; _xj++)
                           {{
                               {body}
                           }}
                       }}
                   }}'''.format(body=self.body)
            clean = '''
                    for (int _xi = 0; _xi < rem; _xi += SOA_SZ)
                    {{
                        for (int _y = 0; _y < _ny; _y++)
                        {{
                            #pragma omp simd
                            for (int _xj = 0; _xj < SOA_SZ; _xj++)
                            {{
                                {body}
                            }}
                        }}
                    }}
                    for (int _xi = (rem/SOA_SZ)*SOA_SZ, _xj = 0; _xj < rem%SOA_SZ; _xj++)
                    {{
                        for (int _y = 0; _y < _ny; _y++)
                        {{
                            {body}
                        }}
                    }}'''.format(body=self.body)

        return '''{spec}
               {{
                   int nblocks = _nx/SZ;
                   int rem = _nx%SZ;
                   #define X_IDX (_xi + _xj)
                   #define X_IDX_AOSOA(v, nv)\
                       ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
                   #pragma omp parallel for
                   for ( int ib = 0; ib < nblocks; ib++ )
                   {{
                       {core}
                   }}
                   int ib = nblocks;
                   {clean}
                   #undef X_IDX
                   #undef X_IDX_AOSOA
               }}'''.format(spec=self._render_spec(), core=core, clean=clean)

    def _render_spec(self):
        # We first need the argument list; starting with the dimensions
        kargs = ['int ' + d for d in self._dims]

        # Now add any scalar arguments
        kargs.extend('{0.dtype} {0.name}'.format(sa) for sa in self.scalargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            # Views
            if va.isview:
                kargs.append('{0.dtype}* __restrict__ {0.name}_v'.format(va))
                kargs.append('const int* __restrict__ {0.name}_vix'
                             .format(va))

                if va.ncdim == 2:
                    kargs.append('const int* __restrict__ {0.name}_vrstri'
                                 .format(va))
            # Arrays
            else:
                # Intent in arguments should be marked constant
                const = 'const' if va.intent == 'in' else ''

                kargs.append('{0} {1.dtype}* __restrict__ {1.name}_v'
                             .format(const, va).strip())

                if self.needs_ldim(va):
                    kargs.append('int ld{0.name}'.format(va))

        return 'void {0}({1})'.format(self.name, ', '.join(kargs))
