# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

struct kargs
{
    int nrow, nblocks;
    fpdtype_t *reduced, *rcurr, *rold;
% if method == 'errest':
    fpdtype_t *rerr, atol, rtol;
% elif method == 'resid' and dt_type == 'matrix':
    fpdtype_t *dt_mat, dt_fac;
% elif method == 'resid':
    fpdtype_t dt_fac;
% endif
};

void reduction(const struct kargs *restrict args)
{
    int nrow = args->nrow, nblocks = args->nblocks;
    fpdtype_t *reduced = args->reduced, *rcurr = args->rcurr, *rold = args->rold;
% if method == 'errest':
    fpdtype_t *rerr = args->rerr, atol = args->atol, rtol = args->rtol;
% elif method == 'resid' and dt_type == 'matrix':
    fpdtype_t *dt_mat = args->dt_mat, dt_fac = args->dt_fac;
% elif method == 'resid':
    fpdtype_t dt_fac = args->dt_fac;
% endif

    #define X_IDX_AOSOA(v, nv) ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)

    // Initalise the reduction variables
    fpdtype_t ${','.join(f'red{i} = 0.0' for i in range(ncola))};

% if norm == 'uniform':
    #pragma omp parallel for reduction(max : ${','.join(f'red{i}' for i in range(ncola))})
% else:
    #pragma omp parallel for reduction(+ : ${','.join(f'red{i}' for i in range(ncola))})
% endif
    for (int ib = 0; ib < nblocks; ib++)
    {
        for (int _y = 0; _y < nrow; _y++)
        {
            for (int _xi = 0; _xi < BLK_SZ; _xi += SOA_SZ)
            {
                #pragma omp simd
                for (int _xj = 0; _xj < SOA_SZ; _xj++)
                {
                    int idx;
                    fpdtype_t temp;

                % for i in range(ncola):
                    idx = _y*BLK_SZ*${ncola} + ib*BLK_SZ*${ncola}*nrow + X_IDX_AOSOA(${i}, ${ncola});

                % if method == 'errest':
                    temp = rerr[idx]/(atol + rtol*max(fabs(rcurr[idx]), fabs(rold[idx])));
                % elif method == 'resid':
                    temp = (rcurr[idx] - rold[idx])/(1.0e-8 + dt_fac${'*dt_mat[idx]' if dt_type == 'matrix' else ''});
                % endif

                % if norm == 'uniform':
                    red${i} = max(red${i}, temp*temp);
                % else:
                    red${i} += temp*temp;
                % endif
                % endfor
                }
            }
        }
    }
    #undef X_IDX_AOSOA

    // Copy
% for i in range(ncola):
    reduced[${i}] = red${i};
% endfor
}
