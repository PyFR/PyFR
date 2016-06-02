# -*- coding: utf-8 -*-

from pyfr.backends.base.generator import BaseKernelGenerator


class OpenMPKernelGenerator(BaseKernelGenerator):
    def render(self):
        # Kernel spec
        spec = self._render_spec()

        if self.ndim == 1:
            tpl = '''
                  {spec}
                  {{
                      #pragma omp parallel
                      {{
                          int align = PYFR_ALIGN_BYTES / sizeof(fpdtype_t);
                          int cb, ce;
                          loop_sched_1d(_nx, align, &cb, &ce);
                          #pragma omp simd
                          for (int _x = cb; _x < ce; _x++)
                          {{
                              {body}
                          }}
                      }}
                  }}'''
        else:
            tpl = '''{spec}
                  {{
                      #pragma omp parallel
                      {{
                          int align = PYFR_ALIGN_BYTES / sizeof(fpdtype_t);
                          int rb, re, cb, ce;
                          loop_sched_2d(_ny, _nx, align, &rb, &re, &cb, &ce);
                          for (int _y = rb; _y < re; _y++)
                          {{
                              #pragma omp simd
                              for (int _x = cb; _x < ce; _x++)
                              {{
                                  {body}
                              }}
                          }}
                      }}
                  }}'''

        return tpl.format(spec=spec, body=self.body)

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

                if va.ncdim >= 1:
                    kargs.append('const int* __restrict__ {0.name}_vcstri'
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

                if self.needs_lsdim(va):
                    kargs.append('int lsd{0.name}'.format(va))

        return 'void {0}({1})'.format(self.name, ', '.join(kargs))
