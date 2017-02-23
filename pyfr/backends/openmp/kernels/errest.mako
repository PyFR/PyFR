# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
errest(int nrow, int ncolb, int ldim, fpdtype_t *__restrict__ error,
       fpdtype_t *__restrict__ x, fpdtype_t *__restrict__ y,
       fpdtype_t *__restrict__ z, fpdtype_t atol, fpdtype_t rtol)
{
    #define X_IDX_AOSOA(v, nv) ((ci/SOA_SZ*(nv) + (v))*SOA_SZ + cj)

    // Initalise the reduction variables
    fpdtype_t ${','.join('err{0} = 0.0'.format(i) for i in range(ncola))};

% if norm == 'uniform':
    #pragma omp parallel reduction(max : ${','.join('err{0}'.format(i) for i in range(ncola))})
% else:
    #pragma omp parallel reduction(+ : ${','.join('err{0}'.format(i) for i in range(ncola))})
% endif
    {
        int align = PYFR_ALIGN_BYTES / sizeof(fpdtype_t);
        int rb, re, cb, ce, idx;
        loop_sched_2d(nrow, ncolb, align, &rb, &re, &cb, &ce);
        int nci = ((ce - cb) / SOA_SZ)*SOA_SZ;

        for (int r = rb; r < re; r++)
        {
            for (int ci = cb; ci < cb + nci; ci += SOA_SZ)
            {
                for (int cj = 0; cj < SOA_SZ; cj++)
                {
                % for i in range(ncola):
                    idx = r*ldim + X_IDX_AOSOA(${i}, ${ncola});
                % if norm == 'uniform':
                    err${i} = max(err${i}, pow(x[idx]/(atol + rtol*max(fabs(y[idx]), fabs(z[idx]))), 2));
                % else:
                    err${i} += pow(x[idx]/(atol + rtol*max(fabs(y[idx]), fabs(z[idx]))), 2);
                % endif
                % endfor
                }
            }

            for (int ci = cb + nci, cj = 0; cj < ce - ci; cj++)
            {
            % for i in range(ncola):
                idx = r*ldim + X_IDX_AOSOA(${i}, ${ncola});
            % if norm == 'uniform':
                err${i} = max(err${i}, pow(x[idx]/(atol + rtol*max(fabs(y[idx]), fabs(z[idx]))), 2));
            % else:
                err${i} += pow(x[idx]/(atol + rtol*max(fabs(y[idx]), fabs(z[idx]))), 2);
            % endif
            % endfor
            }
        }
    }

    // Copy
% for i in range(ncola):
    error[${i}] = err${i};
% endfor
}
