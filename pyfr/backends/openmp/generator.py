# -*- coding: utf-8 -*-

from pyfr.backends.base.generator import BaseKernelGenerator


class OpenMPKernelGenerator(BaseKernelGenerator):
    def render(self):
        if self.ndim == 1:
            inner = '''
                    int cb, ce;
                    loop_sched_1d(_nx, align, &cb, &ce);
                    int nci = ((ce - cb) / SOA_SZ)*SOA_SZ;
                    for (int _xi = cb; _xi < cb + nci; _xi += SOA_SZ)
                    {{
                        #pragma omp simd
                        for (int _xj = 0; _xj < SOA_SZ; _xj++)
                        {{
                            {body}
                        }}
                    }}
                    for (int _xi = cb + nci, _xj = 0; _xj < ce - _xi; _xj++)
                    {{
                        {body}
                    }}'''.format(body=self.body)
        else:
            inner = '''
                    int rb, re, cb, ce;
                    loop_sched_2d(_ny, _nx, align, &rb, &re, &cb, &ce);
                    int nci = ((ce - cb) / SOA_SZ)*SOA_SZ;
                    for (int _y = rb; _y < re; _y++)
                    {{
                        for (int _xi = cb; _xi < cb + nci; _xi += SOA_SZ)
                        {{
                            #pragma omp simd
                            for (int _xj = 0; _xj < SOA_SZ; _xj++)
                            {{
                                {body}
                            }}
                        }}
                        for (int _xi = cb + nci, _xj = 0; _xj < ce - _xi;
                             _xj++)
                        {{
                            {body}
                        }}
                    }}'''.format(body=self.body)

        return '''{spec}
               {{
                   #define X_IDX (_xi + _xj)
                   #define X_IDX_AOSOA(v, nv)\
                       ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
                   int align = PYFR_ALIGN_BYTES / sizeof(fpdtype_t);
                   #pragma omp parallel
                   {{
                       {inner}
                   }}
                   #undef X_IDX
                   #undef X_IDX_AOSOA
               }}'''.format(spec=self._render_spec(), inner=inner)

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
