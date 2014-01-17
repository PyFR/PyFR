# -*- coding: utf-8 -*-

import re

from pyfr.backends.base.generator import BaseKernelGenerator
from pyfr.util import ndrange


class CUDAKernelGenerator(BaseKernelGenerator):
    def __init__(self, *args, **kwargs):
        super(CUDAKernelGenerator, self).__init__(*args, **kwargs)

        # Specialise
        if self.ndim == 1:
            self._dims = ['_nx']
            self._limits = 'if (_x < _nx)'
            self._deref_arg_array = self._deref_arg_array_1d
        else:
            self._dims = ['_ny', '_nx']
            self._limits = 'for (int _y = 0; _y < _ny && _x < _nx; ++_y)'
            self._deref_arg_array = self._deref_arg_array_2d

    def render(self):
        # Get the kernel specification and main body
        spec = self._emit_spec()
        body = self._emit_body()

        # Iteration limits (if statement/for loop)
        limits = self._limits

        # Combine
        return '''{spec}
               {{
                   int _x = blockIdx.x*blockDim.x + threadIdx.x;
                   {limits}
                   {{
                       {body}
                   }}
               }}'''.format(spec=spec, limits=limits, body=body)

    def _emit_spec(self):
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

                # If we are a matrix (ndim = 2) or a non-MPI stacked
                # vector then a leading dimension is required
                if self.ndim == 2 or (va.ncdim > 0 and not va.ismpi):
                    kargs.append('int ld{0.name}'.format(va))

        return '__global__ void {0}({1})'.format(self.name, ', '.join(kargs))

    def _deref_arg(self, arg):
        if arg.isview:
            return self._deref_arg_view(arg)
        else:
            return self._deref_arg_array(arg)

    def _deref_arg_view(self, arg):
        ptns = ['{0}_v[{0}_vix[_x]]',
                r'{0}_v[{0}_vix[_x] + {0}_vcstri[_x]*\1]',
                r'{0}_v[{0}_vix[_x] + {0}_vrstri[_x]*\1 + {0}_vcstri[_x]*\2]']

        return ptns[arg.ncdim].format(arg.name)

    def _deref_arg_array_1d(self, arg):
        # Index expression fragments
        expr = []

        # Leading dimension
        ldim = 'ld' + arg.name if not arg.ismpi else '_nx'

        # Vector n_v[x]
        if arg.ncdim >= 0:
            expr.append('_x')
        # Stacked vector; n_v[ldim*\1 + x]
        if arg.ncdim >= 1:
            expr.append(r'{}*\{}'.format(ldim, arg.ncdim))
        # Doubly stacked vector; n_v[ldim*nv*\1 + ldim*\2 + r]
        if arg.ncdim == 2:
            expr.append(r'{}*{}*\1'.format(ldim, arg.cdims[1]))

        return '{}_v[{}]'.format(arg.name, ' + '.join(expr))

    def _deref_arg_array_2d(self, arg):
        # Index expression fragments
        expr = []

        # Matrix n_v[ldim*_y + _x]
        if arg.ncdim >= 0:
            expr.append('ld{}*_y + _x'.format(arg.name))
        # Stacked matrix; n_v[ldim*_y + \1*_nx + _x]
        if arg.ncdim >= 1:
            expr.append(r'_nx*\{}'.format(arg.ncdim))
        # Doubly stacked matrix; n_v[ldim*_ny*\1 + ldim*_y + \2*_nx + _x]
        if arg.ncdim == 2:
            expr.append(r'ld{}*_ny*\1'.format(arg.name))

        return '{}_v[{}]'.format(arg.name, ' + '.join(expr))

    def _emit_body(self):
        body = self.body
        ptns = [r'\b{0}\b', r'\b{0}\[(\d+)\]', r'\b{0}\[(\d+)\]\[(\d+)\]']

        for va in self.vectargs:
            # Dereference the argument
            darg = self._deref_arg(va)

            # Substitute
            body = re.sub(ptns[va.ncdim].format(va.name), darg, body)

        return body
