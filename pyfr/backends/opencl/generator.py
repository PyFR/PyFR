# -*- coding: utf-8 -*-

import re

from pyfr.backends.base.generator import BaseKernelGenerator


class OpenCLKernelGenerator(BaseKernelGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                   int _x = get_global_id(0);
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
            ka = []

            # Views
            if va.isview:
                ka.append('__global {0.dtype}* restrict {0.name}_v')
                ka.append('__global const int* restrict {0.name}_vix')

                if va.ncdim >= 1:
                    ka.append('__global const int* restrict {0.name}_vcstri')
                if va.ncdim == 2:
                    ka.append('__global const int* restrict {0.name}_vrstri')
            # Arrays
            else:
                if va.intent == 'in':
                    ka.append('__global const {0.dtype}* restrict {0.name}_v')
                else:
                    ka.append('__global {0.dtype}* restrict {0.name}_v')

                # If we are a matrix (ndim = 2) or a non-MPI stacked
                # vector then a leading (sub) dimension is required
                if self.ndim == 2 or (va.ncdim > 0 and not va.ismpi):
                    ka.append('int lsd{0.name}')

            # Format
            kargs.extend(k.format(va) for k in ka)

        return '__kernel void {0}({1})'.format(self.name, ', '.join(kargs))

    def _deref_arg_view(self, arg):
        ptns = ['{0}_v[{0}_vix[_x]]',
                r'{0}_v[{0}_vix[_x] + {0}_vcstri[_x]*\1]',
                r'{0}_v[{0}_vix[_x] + {0}_vrstri[_x]*\1 + {0}_vcstri[_x]*\2]']

        return ptns[arg.ncdim].format(arg.name)

    def _deref_arg_array_1d(self, arg):
        # Leading (sub) dimension
        lsdim = 'lsd' + arg.name if not arg.ismpi else '_nx'

        # Vector name_v[_x]
        if arg.ncdim == 0:
            ix = '_x'
        # Stacked vector; name_v[lsdim*\1 + _x]
        elif arg.ncdim == 1:
            ix = r'{0}*\1 + _x'.format(lsdim)
        # Doubly stacked vector; name_v[(nv*\1 + \2)*lsdim + _x]
        else:
            ix = r'({0}*\1 + \2)*{1} + _x'.format(arg.cdims[1], lsdim)

        return '{0}_v[{1}]'.format(arg.name, ix)

    def _deref_arg_array_2d(self, arg):
        # Matrix name_v[lsdim*_y + _x]
        if arg.ncdim == 0:
            ix = 'lsd{}*_y + _x'.format(arg.name)
        # Stacked matrix; name_v[(_y*nv + \1)*lsdim + _x]
        elif arg.ncdim == 1:
            ix = r'(_y*{0} + \1)*lsd{1} + _x'.format(arg.cdims[0], arg.name)
        # Doubly stacked matrix; name_v[((\1*_ny + _y)*nv + \2)*lsdim + _x]
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
