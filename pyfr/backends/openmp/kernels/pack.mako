<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

struct kargs
{
    ixdtype_t n;
    fpdtype_t *v;
    ixdtype_t *vix, *vrstri;
    fpdtype_t *pmat;
};

void pack_view(const struct kargs *restrict args)
{
    ixdtype_t n = args->n;
    ixdtype_t *vix = args->vix, *vrstri = args->vrstri;
    fpdtype_t *v = args->v, *pmat = args->pmat;

    #pragma omp simd
    for (ixdtype_t i = 0; i < n; i++)
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
