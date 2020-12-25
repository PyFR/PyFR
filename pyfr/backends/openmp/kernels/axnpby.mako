# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
axnpby(int nrow, int ncolb, int ldim,
       ${', '.join('fpdtype_t *__restrict__ x' + str(i) for i in range(nv))},
       ${', '.join('fpdtype_t a' + str(i) for i in range(nv))})
{
    //printf("axnpby start\n");
    //printf("%f %f\n", x0[0], x1[80]);
    //fpdtype_t sum = 0;
    //for ( int i = 0; i < 41; i++ )
    //    sum += x1[i];
    //printf("sum: %f \n", sum);
    int lenAoAoSoA = ncolb/SZ;
    #define X_IDX_AOSOA(v, nv) ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
    #pragma omp parallel for
    for ( int ib = 0; ib < lenAoAoSoA; ib++ )
    {
        int align = PYFR_ALIGN_BYTES / sizeof(fpdtype_t);
        //int rb, re, cb, ce, idx;
        fpdtype_t axn;
        //loop_sched_2d(nrow, ncolb, align, &rb, &re, &cb, &ce);
        //int nci = ((ce - cb) / SOA_SZ)*SOA_SZ;
        int idx;

        //for (int r = rb; r < re; r++)
        for ( int _xi = 0; _xi < SZ; _xi += SOA_SZ )
        {
            //printf("benduduuum\n");
            //for (int ci = cb; ci < cb + nci; ci += SOA_SZ)
            for ( int _y = 0; _y < nrow; _y++ )
            {
                #pragma omp simd
                //for (int cj = 0; cj < SOA_SZ; cj++)
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
    }
    fpdtype_t axn;
    int idx;
    int ib = lenAoAoSoA;
    int rem = ncolb%SZ;
    //for (int r = rb; r < re; r++)
    for ( int _xi = 0; _xi < rem; _xi += SOA_SZ )
    {
        //for (int ci = cb; ci < cb + nci; ci += SOA_SZ)
        for ( int _y = 0; _y < nrow; _y++ )
        {
            #pragma omp simd
            //for (int cj = 0; cj < SOA_SZ; cj++)
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
    //for (int r = rb; r < re; r++)
    for ( int _xi = (rem/SOA_SZ)*SOA_SZ, _xj = 0; _xj < rem%SOA_SZ; _xj++ )
    {
        //for (int ci = cb; ci < cb + nci; ci += SOA_SZ)
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
    //printf("axnpby finished\n");
    //printf("%f %f\n", x0[0], x1[0]);
    for ( int i = 0; i < nrow; i++ )
    {
        //for ( int j = 0; j < ncolb; j++ )
            //printf("%f", x0[i+j]);
        //printf("\n");
    }
}
