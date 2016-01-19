# -*- coding: utf-8 -*-

import re

from pyfr.backends.base.generator import BaseKernelGenerator
from pyfr.util import ndrange


class MICKernelGenerator(BaseKernelGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Specialise
        self._dims = ['_nx'] if self.ndim == 1 else ['_ny', '_nx']

    def render(self):
        # Argument unpacking
        spec, unpack = self._emit_spec_unpack()

        if self.ndim == 1:
            body = self._emit_body_1d()
            return '''
                   void {name}({spec})
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
                   }}'''.format(name=self.name, spec=spec, unpack=unpack,
                                body=body)
        else:
            innerfn = self._emit_inner_func()
            innercall = self._emit_inner_call()
            return '''{innerfn}
                   void {name}({spec})
                   {{
                       {unpack}
                       #pragma omp parallel
                       {{
                           int align = PYFR_ALIGN_BYTES / sizeof(fpdtype_t);
                           int rb, re, cb, ce;
                           loop_sched_2d(_ny, _nx, align, &rb, &re, &cb, &ce);
                           for (int _y = rb; _y < re; _y++)
                           {{
                               {innercall}
                           }}
                       }}
                   }}'''.format(innerfn=innerfn, spec=spec, unpack=unpack,
                                name=self.name, innercall=innercall)

    def _emit_inner_func(self):
        # Get the specification and body
        spec = self._emit_inner_spec()
        body = self._emit_body_2d()

        # Combine
        return '''{spec}
               {{
                   for (int _x = 0; _x < _nx; _x++)
                   {{
                       {body}
                   }}
               }}'''.format(spec=spec, body=body)

    def _emit_inner_call(self):
        # Arguments for the inner function
        iargs = ['ce - cb']
        iargs.extend(sa.name for sa in self.scalargs)

        for va in self.vectargs:
            iargs.extend(self._offset_arg_array_2d(va))

        return '{0}_inner({1});'.format(self.name, ', '.join(iargs))

    def _emit_inner_spec(self):
        # Inner dimension
        ikargs = ['int _nx']

        # Add any scalar arguments
        ikargs.extend('{0.dtype} {0.name}'.format(sa) for sa in self.scalargs)

        # Vector arguments (always arrays as we're 2D)
        for va in self.vectargs:
            const = 'const' if va.intent == 'in' else ''
            stmt = '{0} {1.dtype} *__restrict__ {1.name}_v'.format(const, va)
            stmt = stmt.strip()

            if va.ncdim == 0:
                ikargs.append(stmt)
            else:
                for ij in ndrange(*va.cdims):
                    ikargs.append(stmt + 'v'.join(str(n) for n in ij))

        return ('static PYFR_NOINLINE void {0}_inner({1})'
                .format(self.name, ', '.join(ikargs)))

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

                # If we are a matrix (ndim = 2) or a non-MPI stacked
                # vector then a leading (sub) dimension is required
                if self.ndim == 2 or (va.ncdim > 0 and not va.ismpi):
                    kspec.append('long *arg{0}')
                    kpack.append('int lsd{0.name} = *arg{{0}};'.format(va))

        return (', '.join(a.format(i) for i, a in enumerate(kspec)),
                '\n'.join(a.format(i) for i, a in enumerate(kpack)))

    def _emit_body_1d(self):
        body = self.body
        ptns = [r'\b{0}\b', r'\b{0}\[(\d+)\]', r'\b{0}\[(\d+)\]\[(\d+)\]']

        for va in self.vectargs:
            # Dereference the argument
            darg = self._deref_arg(va)

            # Substitute
            body = re.sub(ptns[va.ncdim].format(va.name), darg, body)

        return body

    def _emit_body_2d(self):
        body = self.body
        ptns = [r'\b{0}\b', r'\b{0}\[(\d+)\]', r'\b{0}\[(\d+)\]\[(\d+)\]']
        subs = ['{0}_v[_x]', r'{0}_v\1[_x]', r'{0}_v\1v\2[_x]']

        for va in self.vectargs:
            body = re.sub(ptns[va.ncdim].format(va.name),
                          subs[va.ncdim].format(va.name), body)

        return body

    def _deref_arg(self, arg):
        if arg.isview:
            ptns = ['{0}_v[{0}_vix[_x]]',
                    r'{0}_v[{0}_vix[_x] + {0}_vcstri[_x]*\1]',
                    r'{0}_v[{0}_vix[_x] + {0}_vrstri[_x]*\1'
                    r' + {0}_vcstri[_x]*\2]']

            return ptns[arg.ncdim].format(arg.name)
        else:
            # Leading (sub) dimension
            lsdim = 'lsd' + arg.name if not arg.ismpi else '_nx'

            # Vector name_v[_x]
            if arg.ncdim == 0:
                ix = '_x'
            # Stacked vector; name_v[lsdim*\1 + _x]
            elif arg.ncdim == 1:
                ix = r'{0}*\1 + _x'.format(lsdim)
            # Doubly stacked vector; name_v[lsdim*nv*\1 + lsdim*\2 + _x]
            else:
                ix = r'{0}*{1}*\1 + {0}*\2 + _x'.format(lsdim, arg.cdims[1])

            return '{0}_v[{1}]'.format(arg.name, ix)

    def _offset_arg_array_2d(self, arg):
        stmts = []

        # Matrix; name + _y*lsdim + cb
        if arg.ncdim == 0:
            stmts.append('{0}_v + _y*lsd{0} + cb'.format(arg.name))
        # Stacked matrix; name + (_y*nv + <0>)*lsdim + cb
        elif arg.ncdim == 1:
            stmts.extend('{0}_v + (_y*{1} + {2})*lsd{0} + cb'
                         .format(arg.name, arg.cdims[0], i)
                         for i in range(arg.cdims[0]))
        # Doubly stacked matrix; name + ((<0>*_ny + _y)*nv + <1>)*lsdim + cb
        else:
            stmts.extend('{0}_v + (({1}*_ny + _y)*{2} + {3})*lsd{0} + cb'
                         .format(arg.name, i, arg.cdims[1], j)
                         for i, j in ndrange(*arg.cdims))

        return stmts
