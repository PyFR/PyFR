# -*- coding: utf-8 -*-

from pyfr.backends.base.generator import BaseKernelGenerator


class OpenCLKernelGenerator(BaseKernelGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Specialise
        if self.ndim == 1:
            self._limits = 'if (_x < _nx)'
        else:
            self._limits = '''
                int _ysize = (_ny + get_local_size(1) - 1) / get_local_size(1);
                int _ystart = get_global_id(1)*_ysize;
                int _yend = (_ystart + _ysize > _ny) ? _ny : _ystart + _ysize;
                for (int _y = _ystart; _x < _nx && _y < _yend; _y++)
            '''

    def render(self):
        spec = self._render_spec()

        return f'''{spec}
               {{
                   int _x = get_global_id(0);
                   #define X_IDX (_x)
                   #define X_IDX_AOSOA(v, nv) SOA_IX(X_IDX, v, nv)
                   #define BLK_IDX 0
                   #define BCAST_BLK(i, ld) i
                   {self._limits}
                   {{
                       {self.body}
                   }}
                   #undef X_IDX
                   #undef X_IDX_AOSOA
                   #undef BLK_IDX
                   #undef BCAST_BLK
               }}'''

    def _render_spec(self):
        # We first need the argument list; starting with the dimensions
        kargs = [f'int {d}' for d in self._dims]

        # Now add any scalar arguments
        kargs.extend(f'{sa.dtype} {sa.name}' for sa in self.scalargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            # Views
            if va.isview:
                kargs.append(f'__global {va.dtype}* restrict {va.name}_v')
                kargs.append(f'__global const int* restrict {va.name}_vix')

                if va.ncdim == 2:
                    kargs.append('__global const int* restrict '
                                 f'{va.name}_vrstri')
            # Arrays
            else:
                if va.intent == 'in':
                    kargs.append(f'__global const {va.dtype}* restrict '
                                 f'{va.name}_v')
                else:
                    kargs.append(f'__global {va.dtype}* restrict {va.name}_v')

                if self.needs_ldim(va):
                    kargs.append(f'int ld{va.name}')

        return '__kernel void {0}({1})'.format(self.name, ', '.join(kargs))
