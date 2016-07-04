# -*- coding: utf-8 -*-

from pyfr.backends.base.generator import BaseKernelGenerator


class OpenCLKernelGenerator(BaseKernelGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Specialise
        if self.ndim == 1:
            self._ix = 'int _x = get_global_id(0);'
            self._limits = 'if (_x < _nx)'
        else:
            self._ix = 'int _x = get_global_id(0), _y = get_global_id(1);'
            self._limits = 'if (_x < _nx && _y < _ny)'

    def render(self):
        # Kernel spec
        spec = self._render_spec()

        # Iteration indicies and limits
        ix, limits = self._ix, self._limits

        # Combine
        return '''{spec}
               {{
                   {ix}
                   #define X_IDX (_x)
                   #define X_IDX_AOSOA(v, nv) SOA_IX(X_IDX, v, nv)
                   {limits}
                   {{
                       {body}
                   }}
                   #undef X_IDX
                   #undef X_IDX_AOSOA
               }}'''.format(spec=spec, ix=ix, limits=limits, body=self.body)

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

                if va.ncdim == 2:
                    ka.append('__global const int* restrict {0.name}_vrstri')
            # Arrays
            else:
                if va.intent == 'in':
                    ka.append('__global const {0.dtype}* restrict {0.name}_v')
                else:
                    ka.append('__global {0.dtype}* restrict {0.name}_v')

                if self.needs_ldim(va):
                    ka.append('int ld{0.name}')

            # Format
            kargs.extend(k.format(va) for k in ka)

        return '__kernel void {0}({1})'.format(self.name, ', '.join(kargs))
