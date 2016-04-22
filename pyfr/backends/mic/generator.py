# -*- coding: utf-8 -*-

import re

from pyfr.backends.base.generator import BaseKernelGenerator
from pyfr.util import ndrange


class MICKernelGenerator(BaseKernelGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Specialise
        if self.ndim == 1:
            self._dims = ['_nx']
            self._deref_arg_array = self._deref_arg_array_1d
        else:
            self._dims = ['_ny', '_nx']
            self._deref_arg_array = self._deref_arg_array_2d

    def render(self):
        # Kernel spec, unpacking code, and body
        spec, unpack = self._emit_spec_unpack()
        body = self._emit_body()

        if self.ndim == 1:
            tpl = '''{spec}
                  {{
                      {unpack}
                      #pragma omp parallel
                      {{
                          int align = PYFR_ALIGN_BYTES / sizeof(fpdtype_t);
                          int cb, ce;
                          loop_sched_1d(_nx, align, &cb, &ce);
                          for (int _x = cb; _x < ce; _x++)
                          {{
                              {body}
                          }}
                      }}
                  }}'''
        else:
            tpl = '''{spec}
                  {{
                      {unpack}
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

        return tpl.format(spec=spec, unpack=unpack, body=body)

    def _emit_spec_unpack(self):
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

                if va.ncdim >= 1:
                    kspec.append('void **arg{0}')
                    kpack.append('const int *{0.name}_vcstri = *arg{{0}};'
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

                if self.needs_lsdim(va):
                    kspec.append('long *arg{0}')
                    kpack.append('int lsd{0.name} = *arg{{0}};'.format(va))

        # Number the arguments
        params = ', '.join(a.format(i) for i, a in enumerate(kspec))
        unpack = '\n'.join(a.format(i) for i, a in enumerate(kpack))

        return 'void {0}({1})'.format(self.name, params), unpack


    def _deref_arg_view(self, arg):
        ptns = ['{0}_v[{0}_vix[_x]]',
                r'{0}_v[{0}_vix[_x] + {0}_vcstri[_x]*\1]',
                r'{0}_v[{0}_vix[_x] + {0}_vrstri[_x]*\1 + {0}_vcstri[_x]*\2]']

        return ptns[arg.ncdim].format(arg.name)

    def _deref_arg_array_1d(self, arg):
        # Leading (sub) dimension
        lsdim = 'lsd' + arg.name if not arg.ismpi else '_nx'

        # Vector: name_v[_x]
        if arg.ncdim == 0:
            ix = '_x'
        # Stacked vector: name_v[lsdim*\1 + _x]
        elif arg.ncdim == 1:
            ix = r'{0}*\1 + _x'.format(lsdim)
        # Doubly stacked vector: name_v[(nv*\1 + \2)*lsdim + _x]
        else:
            ix = r'({0}*\1 + \2)*{1} + _x'.format(arg.cdims[1], lsdim)

        return '{0}_v[{1}]'.format(arg.name, ix)

    def _deref_arg_array_2d(self, arg):
        # Broadcast vector: name_v[_x]
        if arg.isbroadcast:
            ix = '_x'
        # Matrix: name_v[lsdim*_y + _x]
        elif arg.ncdim == 0:
            ix = 'lsd{}*_y + _x'.format(arg.name)
        # Stacked matrix: name_v[(_y*nv + \1)*lsdim + _x]
        elif arg.ncdim == 1:
            ix = r'(_y*{0} + \1)*lsd{1} + _x'.format(arg.cdims[0], arg.name)
        # Doubly stacked matrix: name_v[((\1*_ny + _y)*nv + \2)*lsdim + _x]
        else:
            ix = (r'((\1*_ny + _y)*{0} + \2)*lsd{1} + _x'
                  .format(arg.cdims[1], arg.name))

        return '{0}_v[{1}]'.format(arg.name, ix)

    def _emit_body(self):
        body = self.body
        ptns = [r'\b{0}\b', r'\b{0}\[(\d+)\]', r'\b{0}\[(\d+)\]\[(\d+)\]']

        for va in self.vectargs:
            # Dereference the argument
            if va.isview:
                darg = self._deref_arg_view(va)
            else:
                darg = self._deref_arg_array(va)

            # Substitute
            body = re.sub(ptns[va.ncdim].format(va.name), darg, body)

        return body
