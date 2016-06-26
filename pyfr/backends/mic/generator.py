# -*- coding: utf-8 -*-

from pyfr.backends.base.generator import BaseKernelGenerator


class MICKernelGenerator(BaseKernelGenerator):
    def render(self):
        # Kernel spec and unpacking code
        spec, unpack = self._render_spec_unpack()

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
                   {unpack}
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
               }}'''.format(spec=spec, unpack=unpack, inner=inner)

    def _render_spec_unpack(self):
        # Start by unpacking the dimensions
        kspec = ['long *arg{0}' for d in self._dims]
        kpack = ['int {0} = *arg{{0}};'.format(d) for d in self._dims]

        # Next unpack any scalar arguments
        kspec.extend('double *arg{0}' for sa in self.scalargs)
        kpack.extend('{0.dtype} {0.name} = *arg{{0}};'.format(sa)
                     for sa in self.scalargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            # Views
            if va.isview:
                kspec.append('void **arg{0}')
                kpack.append('{0.dtype} *{0.name}_v = *arg{{0}};'
                             .format(va))

                kspec.append('void **arg{0}')
                kpack.append('const int *{0.name}_vix = *arg{{0}};'
                             .format(va))

                if va.ncdim == 2:
                    kspec.append('void **arg{0}')
                    kpack.append('const int *{0.name}_vrstri = *arg{{0}};'
                                 .format(va))
            # Arrays
            else:
                # Intent in arguments should be marked constant
                const = 'const' if va.intent == 'in' else ''

                kspec.append('void **arg{0}')
                kpack.append('{0} {1.dtype}* {1.name}_v = *arg{{0}};'
                             .format(const, va).strip())

                if self.needs_ldim(va):
                    kspec.append('long *arg{0}')
                    kpack.append('int ld{0.name} = *arg{{0}};'.format(va))

        # Number the arguments
        params = ', '.join(a.format(i) for i, a in enumerate(kspec))
        unpack = '\n'.join(a.format(i) for i, a in enumerate(kpack))

        return 'void {0}({1})'.format(self.name, params), unpack
