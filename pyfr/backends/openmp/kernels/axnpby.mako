# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
axnpby(int nrow, int ncolb, int ldim,
       ${', '.join(f'fpdtype_t *__restrict__ x{i}' for i in range(nv))},
       ${', '.join(f'fpdtype_t a{i}' for i in range(nv))})
{
% if sorted(subdims) == list(range(ncola)):
    #pragma omp parallel
    {
        int align = PYFR_ALIGN_BYTES / sizeof(fpdtype_t);
        int cb, ce;
        loop_sched_1d(nrow*ldim, align, &cb, &ce);

        #pragma omp simd
        for (int i = cb; i < ce; i++)
            x0[i] = ${pyfr.dot('a{l}', 'x{l}[i]', l=nv)};
    }
% else:
    #define X_IDX_AOSOA(v, nv) ((ci/SOA_SZ*(nv) + (v))*SOA_SZ + cj)
    #pragma omp parallel
    {
        int align = PYFR_ALIGN_BYTES / sizeof(fpdtype_t);
        int rb, re, cb, ce, idx;
        loop_sched_2d(nrow, ncolb, align, &rb, &re, &cb, &ce);
        int nci = ((ce - cb) / SOA_SZ)*SOA_SZ;

        for (int r = rb; r < re; r++)
        {
            for (int ci = cb; ci < cb + nci; ci += SOA_SZ)
            {
                #pragma omp simd
                for (int cj = 0; cj < SOA_SZ; cj++)
                {
                % for k in subdims:
                    idx = r*ldim + X_IDX_AOSOA(${k}, ${ncola});

                    x0[idx] = ${pyfr.dot('a{l}', 'x{l}[idx]', l=nv)};
                % endfor
                }
            }

            for (int ci = cb + nci, cj = 0; cj < ce - ci; cj++)
            {
            % for k in subdims:
                idx = r*ldim + X_IDX_AOSOA(${k}, ${ncola});

                x0[idx] = ${pyfr.dot('a{l}', 'x{l}[idx]', l=nv)};
            % endfor
            }
        }
    }
    #undef X_IDX_AOSOA
% endif
}
