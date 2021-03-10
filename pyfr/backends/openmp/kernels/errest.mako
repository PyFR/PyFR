# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
errest(int nrow, int nblocks, fpdtype_t *__restrict__ error,
       fpdtype_t *__restrict__ x, fpdtype_t *__restrict__ y,
       fpdtype_t *__restrict__ z, fpdtype_t atol, fpdtype_t rtol)
{
    #define X_IDX_AOSOA(v, nv) ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)

    // Initalise the reduction variables
    fpdtype_t ${','.join(f'err{i} = 0.0' for i in range(ncola))};

% if norm == 'uniform':
    #pragma omp parallel for reduction(max : ${','.join(f'err{i}' for i in range(ncola))})
% else:
    #pragma omp parallel for reduction(+ : ${','.join(f'err{i}' for i in range(ncola))})
% endif
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
                % for i in range(ncola):
                    idx = _y*BLK_SZ*${ncola} + ib*BLK_SZ*${ncola}*nrow + X_IDX_AOSOA(${i}, ${ncola});
                % if norm == 'uniform':
                    err${i} = max(err${i}, pow(x[idx]/(atol + rtol*max(fabs(y[idx]), fabs(z[idx]))), 2));
                % else:
                    err${i} += pow(x[idx]/(atol + rtol*max(fabs(y[idx]), fabs(z[idx]))), 2);
                % endif
                % endfor
                }
            }
        }
    }
    #undef X_IDX_AOSOA

    // Copy
% for i in range(ncola):
    error[${i}] = err${i};
% endfor
}
