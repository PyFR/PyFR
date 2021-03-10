# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
axnpby(int nrow, int nblocks,
       ${', '.join(f'fpdtype_t *__restrict__ x{i}' for i in range(nv))},
       ${', '.join(f'fpdtype_t a{i}' for i in range(nv))})
{
% if sorted(subdims) == list(range(ncola)):
    #pragma omp parallel for
    for (int ib = 0; ib < nblocks; ib++)
    {
        int idx, blksz = nrow*BLK_SZ*${ncola};
        #pragma omp simd
        for (int i = 0; i < blksz; i++)
        {
            idx = i + ib*blksz;
            x0[idx] = ${pyfr.dot('a{l}', 'x{l}[idx]', l=nv)};
        }
    }
% else:
    #define X_IDX_AOSOA(v, nv) ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
    #pragma omp parallel for
    for (int ib = 0; ib < nblocks; ib++)
    {
        int idx;

        for (int _y = 0; _y < nrow; _y++)
        {
            for (int _xi = 0; _xi < BLK_SZ; _xi += SOA_SZ)
            {
                #pragma omp simd
                for (int _xj = 0; _xj < SOA_SZ; _xj++)
                {
                % for k in subdims:
                    idx = _y*BLK_SZ*${ncola} + ib*BLK_SZ*${ncola}*nrow + X_IDX_AOSOA(${k}, ${ncola});
                    x0[idx] = ${pyfr.dot('a{l}', 'x{l}[idx]', l=nv)};
                % endfor
                }
            }
        }
    }
    #undef X_IDX_AOSOA
% endif
}
