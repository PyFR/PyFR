import re
from math import prod

from pyfr.backends.base.generator import BaseKernelGenerator
from pyfr.util import ndrange


class OpenMPKernelGenerator(BaseKernelGenerator):
    def _render_body_preamble_epilogue(self, body):
        self._has_fp_precise = 'PYFR_FP_PRECISE_BEGIN' in body
        self._staged_reduces = []
        self._bcol_reduces = []
        return self._render_body(body), '', ''

    def _render_reduce(self, va, body, subp, darg):
        # 2D broadcast-col: per-iteration local + direct accumulation
        if va.isbroadcastc:
            self._bcol_reduces.append(va)
            n = prod(va.cdims) if va.cdims else 1

            if va.ncdim:
                decl = f'fpdtype_t {va.name}[{n}];'
                accum = '\n'.join(
                    self._accum_expr(va.reduceop, darg.replace('\\1', str(j)),
                                     f'{va.name}[{j}]')
                    for j in range(n)
                )
            else:
                decl = f'fpdtype_t {va.name};'
                accum = self._accum_expr(va.reduceop, darg, va.name)

            return f'{decl}\n{body}\n{accum}'
        # 1D view: staging array + atomic writeback
        else:
            self._staged_reduces.append(va)
            vs = va.viewstride

            if vs > 1:
                ptn = r'_rv_{0}[\1][X_IDX]'.format(va.name)
            else:
                ptn = r'_rv_{0}[X_IDX]'.format(va.name)

            return re.sub(subp, ptn, body)

    def _render_staging(self):
        # Generate staging array declarations and atomic writeback code
        decls, atoms = [], []
        ix = '_xi + _xj'

        for va in self._staged_reduces:
            afn = f'atomic_{va.reduceop}_fpdtype'
            name = va.name
            vs = va.viewstride

            if vs > 1:
                decls.append(f'fpdtype_t _rv_{name}[{vs}][BLK_SZ];')
                for i in range(vs):
                    vidx = f'{vs}*({ix}) + {i}'
                    atoms.append(
                        f'{afn}(&{name}_v[{name}_vix[{vidx}]], '
                        f'_rv_{name}[{i}][{ix}]);')
            else:
                decls.append(f'fpdtype_t _rv_{name}[BLK_SZ];')
                atoms.append(f'{afn}(&{name}_v[{name}_vix[{ix}]], '
                             f'_rv_{name}[{ix}]);')

        return '\n'.join(decls), '\n'.join(atoms)

    def _render_bcol_reduce_init(self):
        # Identity-initialise broadcast-col reduce outputs
        lines = []
        for va in self._bcol_reduces:
            ident = self._reduce_ident(va.reduceop)
            n = prod(va.cdims) if va.cdims else 1
            sz = f'{{0}}*{n}' if n > 1 else '{0}'
            lines.append(f'for (int _i = 0; _i < {sz}; _i++) '
                         f'{va.name}_v[_i] = {ident};')

        return '\n'.join(lines)

    def render(self):
        kargdefn, kargassn = self._render_args('args')
        rdecls, rflush = self._render_staging()

        # View reduce: atomic flush after each SIMD chunk
        if rflush:
            rflush_core = (f'for (int _xj = 0; _xj < SOA_SZ; _xj++) '
                           f'{{ {rflush} }}')
            rflush_clean = (f'for (int _xj = 0; _xj < _nx % BLK_SZ; _xj++) '
                            f'{{ {rflush} }}')
        else:
            rflush_core = rflush_clean = ''

        # Broadcast-col reduce identity init
        rinit = self._render_bcol_reduce_init()
        rinit_core = rinit.format('BLK_SZ')
        rinit_clean = rinit.format('_nx % BLK_SZ')

        if self.ndim == 1:
            core = f'''
                for (int _xi = 0; _xi < BLK_SZ; _xi += SOA_SZ)
                {{
                    #pragma omp simd
                    for (int _xj = 0; _xj < SOA_SZ; _xj++)
                    {{
                        {self.body}
                    }}
                    {rflush_core}
                }}'''
            clean = f'''
                int _xi = 0;
                #pragma omp simd
                for (int _xj = 0; _xj < _nx % BLK_SZ; _xj++)
                {{
                    {self.body}
                }}
                {rflush_clean}'''
        else:
            core = f'''
                {rinit_core}
                for (ixdtype_t _y = 0; _y < _ny; _y++)
                {{
                    for (int _xi = 0; _xi < BLK_SZ; _xi += SOA_SZ)
                    {{
                        #pragma omp simd
                        for (int _xj = 0; _xj < SOA_SZ; _xj++)
                        {{
                            {self.body}
                        }}
                        {rflush_core}
                    }}
                }}'''
            clean = f'''
                {rinit_clean}
                for (ixdtype_t _y = 0, _xi = 0; _y < _ny; _y++)
                {{
                    #pragma omp simd
                    for (int _xj = 0; _xj < _nx % BLK_SZ; _xj++)
                    {{
                        {self.body}
                    }}
                    {rflush_clean}
                }}'''

        result = f'''
            struct {self.name}_kargs {{ {kargdefn}; }};
            void {self.name}(ixdtype_t _ib,
                             const struct {self.name}_kargs *args,
                             int _disp_mask)
            {{
                {kargassn};
                {rdecls}
                #define X_IDX (_xi + _xj)
                #define X_IDX_AOSOA(v, nv)\
                    ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
                #define BCAST_BLK(r, c, ld) ((c) % (ld) + ((c) / (ld))*(ld)*r)
                if (_nx - _ib*BLK_SZ >= BLK_SZ)
                {{
                    {core}
                }}
                else
                {{
                    {clean}
                }}
                #undef X_IDX
                #undef X_IDX_AOSOA
                #undef BCAST_BLK
            }}'''

        if self._has_fp_precise:
            result = '// PYFR_DISABLE_FAST_MATH\n' + result

        return result

    def ldim_size(self, name, factor=1):
        return f'{factor}*BLK_SZ' if factor > 1 else 'BLK_SZ'

    def needs_ldim(self, arg):
        return False

    def _displace_arg(self, arg):
        if arg.isview:
            return None
        elif self.ndim == 1:
            # Vector
            if arg.ncdim == 0 or arg.ismpi:
                return '_ib*BLK_SZ'
            # 2D broadcast vector
            elif arg.isbroadcast:
                return None
            # Stacked vector:
            else:
                return f'_ib*BLK_SZ*{prod(arg.cdims)}'
        else:
            # 2D broadcast vector or row broadcast matrix
            if arg.isbroadcast or arg.isbroadcastr:
                return None
            # Column broadcast matrix
            elif arg.isbroadcastc:
                return f'_ib*BLK_SZ*{prod(arg.cdims)}'
            # Matrix
            else:
                return f'_ib*BLK_SZ*{prod(arg.cdims)}*_ny'

    def _render_args(self, argn):
        # We first need the argument list; starting with the dimensions
        kargs = [('ixdtype_t', d, None, None) for d in self._dims]

        # Now add any scalar arguments
        kargs.extend((sa.dtype, sa.name, None, None) for sa in self.scalargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            da = self._displace_arg(va)
            mi = len(kargs) if da else None

            if va.intent == 'in':
                kargs.append((f'const {va.dtype}*', f'{va.name}_v', da, mi))
            else:
                kargs.append((f'{va.dtype}*', f'{va.name}_v', da, mi))

            # Views
            if va.isview:
                vix_disp = f'_ib*BLK_SZ*{va.viewstride}'
                kargs.append(('const ixdtype_t*', f'{va.name}_vix', vix_disp,
                              None))

                if self.ndim == 2 and not va.isbroadcastc:
                    kargs.append(('ixdtype_t', f'{va.name}_vrstri', None,
                                  None))
                elif va.ncdim == 2:
                    kargs.append(('const ixdtype_t*', f'{va.name}_vrstri',
                                  '_ib*BLK_SZ', None))

        # Argument definitions and assignments
        kargdefn, kargassn = [], []
        for dtype, name, disp, midx in kargs:
            assn = f'{dtype} {name} = {argn}->{name}'

            # Handle displacement and potential masking thereof
            if disp and midx is not None:
                assn += f' + ((_disp_mask & {1 << midx}) ? 0 : {disp})'
            elif disp:
                assn += f' + {disp}'

            kargdefn.append(f'{dtype} {name}')
            kargassn.append(assn)

        return ';\n'.join(kargdefn), ';\n'.join(kargassn)
