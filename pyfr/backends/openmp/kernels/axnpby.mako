# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
axnpby(int nrow, int nblocks,
       ${', '.join(f'fpdtype_t *restrict x{i}' for i in range(nv))},
       ${', '.join(f'fpdtype_t a{i}' for i in range(nv))})
{
% if sorted(subdims) == list(range(ncola)):
    #pragma omp parallel for
    for (int ib = 0; ib < nblocks; ib++)
    {
        #pragma omp simd
        for (int i = ib*nrow*BLK_SZ*${ncola}; i < (ib + 1)*nrow*BLK_SZ*${ncola}; i++)
            x0[i] = ${pyfr.dot('a{l}', 'x{l}[i]', l=nv)};
    }
% else:
    #define X_IDX_AOSOA(v, nv) ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
    #pragma omp parallel for
    for (int ib = 0; ib < nblocks; ib++)
    {
        for (int _y = 0; _y < nrow; _y++)
        {
            for (int _xi = 0; _xi < BLK_SZ; _xi += SOA_SZ)
            {
                #pragma omp simd
                for (int _xj = 0; _xj < SOA_SZ; _xj++)
                {
                    int i;

                % for k in subdims:
                    i = _y*BLK_SZ*${ncola} + ib*BLK_SZ*${ncola}*nrow + X_IDX_AOSOA(${k}, ${ncola});
                    x0[i] = ${pyfr.dot('a{l}', 'x{l}[i]', l=nv)};
                % endfor
                }
            }
        }
    }
    #undef X_IDX_AOSOA
% endif
}
