# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
axnpby(int nrow, int ncolb, int ldim,
       ${', '.join(f'fpdtype_t *__restrict__ x{i}' for i in range(nv))},
       ${', '.join(f'fpdtype_t a{i}' for i in range(nv))})
{
% if sorted(subdims) == list(range(ncola)):
    int nblocks = (ncolb - ncolb%-SZ)/SZ;
    #pragma omp parallel for
    for ( int ib = 0; ib < nblocks; ib++ )
    {
        int idx, blksz = nrow*SZ*${ncola};
        #pragma omp simd
        for ( int i = 0; i < blksz; i++ )
        {
            idx = i + ib*blksz;
            x0[idx] = ${pyfr.dot('a{l}', 'x{l}[idx]', l=nv)};
        }
    }
% else:
    int nblocks = ncolb/SZ;
    #define X_IDX_AOSOA(v, nv) ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
    #pragma omp parallel for
    for ( int ib = 0; ib < nblocks; ib++ )
    {
        int idx;

        for ( int _xi = 0; _xi < SZ; _xi += SOA_SZ )
        {
            for ( int _y = 0; _y < nrow; _y++ )
            {
                #pragma omp simd
                for ( int _xj = 0; _xj < SOA_SZ; _xj++ )
                {
                % for k in subdims:
                    idx = _y*ldim + ib*SZ*${ncola}*nrow + X_IDX_AOSOA(${k}, ${ncola});
                    x0[idx] = ${pyfr.dot('a{l}', 'x{l}[idx]', l=nv)};
                % endfor
                }
            }
        }
    }
    int idx;
    int ib = nblocks;
    int rem = ncolb%SZ;
    for ( int _xi = 0; _xi < rem; _xi += SOA_SZ )
    {
        for ( int _y = 0; _y < nrow; _y++ )
        {
            #pragma omp simd
            for ( int _xj = 0; _xj < SOA_SZ; _xj++)
            {
            % for k in subdims:
                idx = _y*ldim + ib*SZ*${ncola}*nrow + X_IDX_AOSOA(${k}, ${ncola});
                x0[idx] = ${pyfr.dot('a{l}', 'x{l}[idx]', l=nv)};
            % endfor
            }
        }
    }
    for ( int _xi = (rem/SOA_SZ)*SOA_SZ, _xj = 0; _xj < rem%SOA_SZ; _xj++ )
    {
        for ( int _y = 0; _y < nrow; _y++ )
        {
        % for k in subdims:
            idx = _y*ldim + ib*SZ*${ncola}*nrow + X_IDX_AOSOA(${k}, ${ncola});
            x0[idx] = ${pyfr.dot('a{l}', 'x{l}[idx]', l=nv)};
        % endfor
        }
    }
    #undef X_IDX_AOSOA
% endif
}
