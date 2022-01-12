# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

struct kargs
{
    int n;
    fpdtype_t *v;
    int *vix, *vrstri;
    fpdtype_t *pmat;
};

void pack_view(const struct kargs *restrict args)
{
    int n = args->n;
    int *vix = args->vix, *vrstri = args->vrstri;
    fpdtype_t *v = args->v, *pmat = args->pmat;

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
