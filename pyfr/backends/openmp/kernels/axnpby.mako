# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
axnpby(int nrow, int ncolb, int ldim,
       ${', '.join('fpdtype_t *__restrict__ x' + str(i) for i in range(nv))},
       ${', '.join('fpdtype_t a' + str(i) for i in range(nv))})
{
    int nblocks = ncolb/SZ;
    #define X_IDX_AOSOA(v, nv) ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
    #pragma omp parallel for
    for ( int ib = 0; ib < nblocks; ib++ )
    {
        fpdtype_t axn;
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
                    axn = ${pyfr.dot('a{l}', 'x{l}[idx]', l=(1, nv))};

                    if (a0 == 0.0)
                        x0[idx] = axn;
                    else if (a0 == 1.0)
                        x0[idx] += axn;
                    else
                        x0[idx] = a0*x0[idx] + axn;
                % endfor
                }
            }
        }
    }
    fpdtype_t axn;
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
                axn = ${pyfr.dot('a{l}', 'x{l}[idx]', l=(1, nv))};

                if (a0 == 0.0)
                    x0[idx] = axn;
                else if (a0 == 1.0)
                    x0[idx] += axn;
                else
                    x0[idx] = a0*x0[idx] + axn;
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
            axn = ${pyfr.dot('a{l}', 'x{l}[idx]', l=(1, nv))};

            if (a0 == 0.0)
                x0[idx] = axn;
            else if (a0 == 1.0)
                x0[idx] += axn;
            else
                x0[idx] = a0*x0[idx] + axn;
        % endfor
        }
    }
    #undef X_IDX_AOSOA
}
