# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
pack_view(int n,
          const fpdtype_t *__restrict__ v,
          const int *__restrict__ vix,
          const int *__restrict__ vrstri,
          fpdtype_t *__restrict__ pmat)
{
    #pragma omp simd
    for (int i = 0; i < n; i++)
    {
    % if nrv == 1:
    % for c in range(ncv):
        pmat[${c}*n + i] = v[vix[i] + SOA_SZ*${c}];
    % endfor
    % else:
    % for r, c in pyfr.ndrange(nrv, ncv):
        pmat[${r*ncv + c}*n + i] = v[vix[i] + vrstri[i]*${r} + SOA_SZ*${c}];
    % endfor
    % endif
    }
}
