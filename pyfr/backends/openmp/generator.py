# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base.generator import (BaseKernelGenerator,
                                          BaseFunctionGenerator, funccall,
                                          funcsig)
from pyfr.util import ndrange


class OpenMPKernelGenerator(BaseKernelGenerator):
    def __init__(self, *args, **kwargs):
        super(OpenMPKernelGenerator, self).__init__(*args, **kwargs)

        # Specialise
        if self.ndim == 1:
            self._outerit, self._nouter = '_x', '_nx'
            self._dims = [self._nouter]

            self._emit_load_store = self._emit_load_store_1d
            self._emit_outer_loop_body = self._emit_outer_loop_body_1d
        else:
            self._outerit, self._nouter = '_y', '_ny'
            self._innerit, self._ninner = '_x', '_nx'
            self._dims = [self._nouter, self._ninner]

            self._emit_load_store = self._emit_load_store_2d
            self._emit_outer_loop_body = self._emit_outer_loop_body_2d

    def render(self):
        # An OpenMP kernel takes the general form of:
        #   InnerFunc (2D only)         | Inner-loop function; invoked
        #                               | by the body of the outer loop
        #   void                        |
        #   Name(Arguments)             | Outer function spec
        #   {                           |
        #     #pragma omp parallel for  |
        #     for (...)                 | Outer loop
        #         ...                   | Outer loop body
        #   }                           |
        spec = self._emit_outer_spec()
        head = self._emit_outer_loop()
        body = self._emit_outer_loop_body()

        # Fix indentation
        head = [' '*4 + l for l in head]
        body = [' '*8 + l for l in body]

        # Add in the missing '{' and '}' to form the kernel
        body = ['{'] + head + ['    {'] + body + ['    }', '}']

        # Combine to yield the outer kernel
        kern = spec + body

        # In 2D we need to bring in an inner function
        if self.ndim == 2:
            kern = self._emit_inner_func() + [''] + kern

        # Flattern
        return '\n'.join(kern)

    def _emit_inner_func(self):
        # Inner functions, which contain the inner loop in 2D kernels,
        # take the form of:
        #   static NOINLINE void        |
        #   InnerName(InnerArguments)   | Inner function prototype
        #   {                           |
        #     Align specs               | Argument aligns
        #                               |
        #     for (...)                 | Inner loop
        #     {                         |
        #       Arg decl                | Local argument declarations
        #       In arg loads            | Global-to-local argument loads
        #                               |
        #       Main body               | Pointwise op
        #                               |
        #       Out arg stores          | Local-to-global argument stores
        #     }                         |
        #   }
        spec = self._emit_inner_spec()

        head = self._emit_inner_aligns() + ['']
        head += self._emit_inner_loop()

        body = self._emit_arg_decls() + ['']
        body += self._emit_assigments('in')
        body += self.body
        body += self._emit_assigments('out')

        # Fix indentation
        head = [' '*4 + l for l in head]
        body = [' '*8 + l for l in body]

        # Add in the missing '{' and '}' to form the inner function
        body = ['{'] + head + ['    {'] + body + ['    }', '}']

        # Combine
        return spec + body

    def _emit_outer_loop_body_1d(self):
        body = self._emit_arg_decls() + ['']
        body += self._emit_assigments('in')
        body += self.body
        body += self._emit_assigments('out')

        return body

    def _emit_outer_loop_body_2d(self):
        # Arguments for the inner function
        iargs = [self._ninner]
        iargs.extend(sa.name for sa in self.scalargs)

        for va in self.vectargs:
            iargs.extend(self._offset_arg_array_2d(va))

        fcall = funccall(self.name + '_inner', iargs)

        # Terminate the line
        fcall[-1] += ';'

        return fcall

    def _emit_inner_spec(self):
        # Inner dimension
        ikargs = ['int ' + self._ninner]

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

        return funcsig('static PYFR_NOINLINE', 'void', self.name + '_inner',
                       ikargs)

    def _emit_outer_spec(self):
        # We first need the argument list; starting with the dimensions
        kargs = ['int ' + d for d in self._dims]

        # Now add any scalar arguments
        kargs.extend('{0.dtype} {0.name}'.format(sa) for sa in self.scalargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            # Views
            if va.isview:
                kargs.append('{0.dtype}** __restrict__ {0.name}_v'
                             .format(va))
                kargs.append('const int* __restrict__ {0.name}_vstri'
                             .format(va))
            # Arrays
            else:
                # Intent in arguments should be marked constant
                const = 'const' if va.intent == 'in' else ''

                kargs.append('{0} {1.dtype}* __restrict__ {1.name}_v'
                             .format(const, va).strip())

                # If we are a matrix (ndim = 2) or a non-MPI stacked
                # vector then a leading (sub) dimension is required
                if self.ndim == 2 or (va.ncdim > 0 and not va.ismpi):
                    kargs.append('int lsd{0.name}'.format(va))

        return funcsig('', 'void', self.name, kargs)

    def _emit_inner_aligns(self):
        aligns = []
        for va in self.vectargs:
            if va.ncdim == 0:
                aligns.append('PYFR_ALIGNED({0.name}_v);'.format(va))
            else:
                aligns.extend('PYFR_ALIGNED({0.name}_v{1});'
                              .format(va, 'v'.join(str(n) for n in ij))
                              for ij in ndrange(*va.cdims))

        return aligns

    def _emit_for(self, it, n, openmp=True):
        loop = 'for (int {0} = 0; {0} < {1}; {0}++)'.format(it, n)

        if openmp:
            return ['#pragma omp parallel for', loop]
        else:
            return [loop]

    def _emit_inner_loop(self):
        return self._emit_for(self._innerit, self._ninner, openmp=False)

    def _emit_outer_loop(self):
        return self._emit_for(self._outerit, self._nouter, openmp=True)

    def _emit_assigments(self, intent):
        assigns = []
        for va in self.vectargs:
            if intent in va.intent and va.isused:
                lg = self._emit_load_store(va)

                if intent == 'in':
                    assigns.append('// Load ' + va.name)
                    assigns.extend('{} = {};'.format(l, g) for l, g in lg)
                else:
                    assigns.append('// Store ' + va.name)
                    assigns.extend('{} = {};'.format(g, l) for l, g in lg)

                assigns.append('')

        # Strip the trailing '' and return
        return assigns[:-1]

    def _emit_load_store_1d(self, arg):
        # Dereference the argument
        darg = self._deref_arg(arg)

        if arg.ncdim == 0:
            return [(arg.name, darg)]
        else:
            exprs = []
            for ij in ndrange(*arg.cdims):
                # Local variable; name[<i>] or name[<i>][<j>]
                lidx = '[{}]'.format(']['.join(str(n) for n in ij))
                lvar = arg.name + lidx

                # Global variable; varies
                gvar = darg.format(*ij)

                exprs.append((lvar, gvar))

            return exprs

    def _emit_load_store_2d(self, arg):
        if arg.ncdim == 0:
            return [(arg.name, arg.name + '_v[{}]'.format(self._innerit))]
        else:
            exprs = []
            for ij in ndrange(*arg.cdims):
                # Local variable; name[<i>] or name[<i>][<j>]
                lidx = '[{}]'.format(']['.join(str(n) for n in ij))
                lvar = arg.name + lidx

                # Global variable; name_v<i>[ix] or name_v<i>v<j>[ix]
                gvar = '{0}_v{2}[{1}]'.format(arg.name, self._innerit,
                                              'v'.join(str(n) for n in ij))

                exprs.append((lvar, gvar))

            return exprs

    def _emit_arg_decls(self):
        decls = []
        for va in self.vectargs:
            if va.isused:
                if va.ncdim == 0:
                    decls.append('{0.dtype} {0.name};'.format(va))
                else:
                    arr = ']['.join(str(d) for d in va.cdims)
                    decls.append('{0.dtype} {0.name}[{1}];'.format(va, arr))

        return decls

    def _deref_arg(self, arg):
        if arg.isview:
            return self._deref_arg_view(arg)
        else:
            return self._deref_arg_array(arg)

    def _deref_arg_view(self, arg):
        ncdim = len(arg.cdims)
        r, nr = self._outerit, self._nouter

        if arg.ncdim == 1:
            expr, cidx = r, '{0}'
        elif arg.ncdim == 2:
            expr, cidx = '{{0}}*{} + {}'.format(nr, r), '{1}'

        return '{0}_v[{1}][{0}_vstri[{1}]*{2}]'.format(arg.name, expr, cidx)

    def _deref_arg_array(self, arg):
        # Index expression fragments
        expr = []

        # Leading dimension
        ldim = 'lsd' + arg.name if not arg.ismpi else self._nouter

        # Vector n_v[x]
        if arg.ncdim >= 0:
            expr.append(self._outerit)
        # Stacked vector; n_v[ldim*<0> + x]
        if arg.ncdim >= 1:
            expr.append('{}*{}'.format(ldim,
                                       '{0}' if arg.ncdim == 1 else '{1}'))
        # Doubly stacked vector; n_v[ldim*nv*<0> + ldim*<1> + r]
        if arg.ncdim == 2:
            expr.append('{}*{}*{{0}}'.format(ldim, arg.cdims[1]))

        return '{}_v[{}]'.format(arg.name, ' + '.join(expr))

    def _offset_arg_array_2d(self, arg):
        r, nr = self._outerit, self._nouter
        stmts = []

        # Matrix; name + r*lsdim
        if arg.ncdim == 0:
            stmts.append('{0}_v + {1}*lsd{0}'.format(arg.name, r))
        # Stacked matrix; name + (r*nv + <0>)*lsdim
        elif arg.ncdim == 1:
            stmts.extend('{0}_v + ({1}*{2} + {3})*lsd{0}'
                         .format(arg.name, r, arg.cdims[0], i)
                         for i in range(arg.cdims[0]))
        # Doubly stacked matrix; name + ((<0>*nr + r)*nv + <1>)*lsdim
        else:
            stmts.extend('{0}_v + (({1}*{2} + {3})*{4} + {5})*lsd{0}'
                         .format(arg.name, i, nr, r, arg.cdims[1], j)
                         for i, j in ndrange(*arg.cdims))

        return stmts

    def argspec(self):
        # Argument names and types
        argn, argt = [], []

        # Dimensions
        argn += self._dims
        argt += [[np.int32]]*self.ndim

        # Scalar args (always of type fpdtype)
        argn += [sa.name for sa in self.scalargs]
        argt += [[self.fpdtype]]*len(self.scalargs)

        # Vector args
        for va in self.vectargs:
            argn.append(va.name)

            # View
            if va.isview:
                argt.append([np.intp, np.intp])
            # Non-stacked vector or MPI type
            elif self.ndim == 1 and (va.ncdim == 0 or va.ismpi):
                argt.append([np.intp])
            # Stacked vector/matrix/stacked matrix
            else:
                argt.append([np.intp, np.int32])

        # Return
        return self.ndim, argn, argt


class OpenMPFunctionGenerator(BaseFunctionGenerator):
    @property
    def mods(self):
        return 'static inline'
