# -*- coding: utf-8 -*-

from pyfr.backends.base.generator import BaseKernelGenerator


class OpenCLKernelGenerator(BaseKernelGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Specialise
        if self.ndim == 1:
            self._limits = 'if (_x < _nx)'
        else:
            self._limits = 'for (int _y = 0; _y < _ny && _x < _nx; ++_y)'

    def render(self):
        # Kernel spec
        spec = self._render_spec()

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
               }}'''.format(spec=spec, limits=limits, body=self.body)

    def _render_spec(self):
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

                if self.needs_lsdim(va):
                    ka.append('int lsd{0.name}')

            # Format
            kargs.extend(k.format(va) for k in ka)

        return '__kernel void {0}({1})'.format(self.name, ', '.join(kargs))
