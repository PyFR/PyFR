# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base.generator import (BaseKernelGenerator,
                                          BaseFunctionGenerator, funcsig)
from pyfr.util import ndrange


class CUDAKernelGenerator(BaseKernelGenerator):
    _xidx, _yidx = '_x', '_y'
    _xdim, _ydim = '_nx', '_ny'

    def __init__(self, *args, **kwargs):
        super(CUDAKernelGenerator, self).__init__(*args, **kwargs)

        # Specialise
        if self.ndim == 1:
            self._dims = [self._xdim]
            self._emit_itidx_check = self._emit_itidx_check_1d
            self._deref_arg_array = self._deref_arg_array_1d
        else:
            self._dims = [self._ydim, self._xdim]
            self._emit_itidx_check = self._emit_itidx_check_2d
            self._deref_arg_array = self._deref_arg_array_2d

    def render(self):
        # A CUDA kernel takes the general form of:
        #   __global__ void      |
        #   Name(Arguments)      | Prototype
        #   {                    |
        #     int _x = ...       | Iteration indices
        #                        |
        #     if/for (...)       | Iteration loop/bounds checks
        #     {                  |
        #       Arg decl         | Local argument declarations
        #       In arg loads     | Global-to-local argument loads
        #                        |
        #       Main body        | Pointwise op
        #                        |
        #       Out arg stores   | Local-to-global argument stores
        #     }                  |
        #   }                    |
        proto = self._emit_prototype()

        head = self._emit_itidx() + ['']
        head += self._emit_itidx_check()

        body = self._emit_arg_decls() + ['']
        body += self._emit_assignments('in')
        body += self.body
        body += self._emit_assignments('out')

        # Fix indentation
        head = [' '*4 + l for l in head]
        body = [' '*8 + l for l in body]

        # Add in the missing '{' and '}' to form the kernel
        body = ['{'] + head + ['    {'] + body + ['    }', '}']

        # Combine
        return '\n'.join(proto + body)

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
                argt.append([np.intp, np.intp, np.intp])
            # Non-stacked vector or MPI type
            elif self.ndim == 1 and (va.ncdim == 0 or va.ismpi):
                argt.append([np.intp])
            # Stacked vector/matrix/stacked matrix
            else:
                argt.append([np.intp, np.int32])

        # Return
        return self.ndim, argn, argt

    def _emit_prototype(self):
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
                kargs.append('const int* __restrict__ {0.name}_vstri'
                             .format(va))
            # Arrays
            else:
                # Intent in arguments should be marked constant
                const = 'const' if va.intent == 'in' else ''

                kargs.append('{0} {1.dtype}* __restrict__ {1.name}_v'
                             .format(const, va).strip())

                # If we are a matrix (ndim = 2) or a non-MPI stacked
                # vector then a leading dimension is required
                if self.ndim == 2 or (va.ncdim > 0 and not va.ismpi):
                    kargs.append('int ld{0.name}'.format(va))

        return funcsig('__global__', 'void', self.name, kargs)

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

    def _emit_assignments(self, intent):
        exprs = []
        for va in self.vectargs:
            if intent in va.intent and va.isused:
                ls = self._emit_load_store(va)

                if intent == 'in':
                    exprs.append('// Load ' + va.name)
                    exprs.extend('{} = {};'.format(l, g) for l, g in ls)
                else:
                    exprs.append('// Store ' + va.name)
                    exprs.extend('{} = {};'.format(g, l) for l, g in ls)

                exprs.append('')

        # Strip the trailing '' and return
        return exprs[:-1]

    def _deref_arg(self, arg):
        if arg.isview:
            return self._deref_arg_view(arg)
        else:
            return self._deref_arg_array(arg)

    def _deref_arg_view(self, arg):
        ncdim = len(arg.cdims)
        r, nr = self._xidx, self._xdim

        if arg.ncdim == 1:
            expr, cidx = r, '{0}'
        elif arg.ncdim == 2:
            expr, cidx = '{{0}}*{} + {}'.format(nr, r), '{1}'

        return ('{0}_v[{0}_vix[{1}] + {0}_vstri[{1}]*{2}]'
                .format(arg.name, expr, cidx))

    def _emit_load_store(self, arg):
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

    def _emit_itidx(self):
        return ['int {} = blockIdx.x*blockDim.x + threadIdx.x;'
                .format(self._xidx)]

    def _emit_itidx_check_1d(self):
        return ['if ({} < {})'.format(self._xidx, self._xdim)]

    def _emit_itidx_check_2d(self):
        return ['for (int {0} = 0; {0} < {1} && {2} < {3}; ++{0})'
                .format(self._yidx, self._ydim, self._xidx, self._xdim)]

    def _deref_arg_array_1d(self, arg):
        # Index expression fragments
        expr = []

        # Leading dimension
        ldim = 'ld' + arg.name if not arg.ismpi else self._xdim

        # Vector n_v[x]
        if arg.ncdim >= 0:
            expr.append(self._xidx)
        # Stacked vector; n_v[ldim*<0> + x]
        if arg.ncdim >= 1:
            expr.append('{}*{}'.format(ldim,
                                       '{0}' if arg.ncdim == 1 else '{1}'))
        # Doubly stacked vector; n_v[ldim*nv*<0> + ldim*<1> + r]
        if arg.ncdim == 2:
            expr.append('{}*{}*{{0}}'.format(ldim, arg.cdims[1]))

        return '{}_v[{}]'.format(arg.name, ' + '.join(expr))

    def _deref_arg_array_2d(self, arg):
        c, r = self._xidx, self._yidx
        nc, nr = self._xdim, self._ydim

        # Index expression fragments
        expr = []

        # Matrix n_v[ldim*r + c]
        if arg.ncdim >= 0:
            expr.append('ld{}*{}'.format(arg.name, r))
            expr.append(c)
        # Stacked matrix; n_v[r*ldim + <0>*nc + c]
        if arg.ncdim >= 1:
            expr.append('{}*{}'.format(nc, '{0}' if arg.ncdim == 1 else '{1}'))
        # Doubly stacked matrix; n_v[ldim*nr*<0> + ldim*r + <1>*nc + c]
        if arg.ncdim == 2:
            expr.append('ld{}*{}*{{0}}'.format(arg.name, nr))

        return '{}_v[{}]'.format(arg.name, ' + '.join(expr))


class CUDAFunctionGenerator(BaseFunctionGenerator):
    @property
    def mods(self):
        return 'static inline __device__'
