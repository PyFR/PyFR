# -*- coding: utf-8 -*-
<%inherit file='base'/>

void
pack_view(long *n_a, long *nrv_a, long *ncv_a,
          void **v_a, void **vix_a, void **vrstri_a,
          void **pmat_a)
{
    int n = *n_a;
    int nrv = *nrv_a;
    int ncv = *ncv_a;

    fpdtype_t *v = *v_a;
    int *vix = *vix_a;
    int *vrstri = (vrstri_a) ? *vrstri_a : 0;
    fpdtype_t *pmat = *pmat_a;

    if (ncv == 1)
        for (int i = 0; i < n; i++)
            pmat[i] = v[vix[i]];
    else if (nrv == 1)
        for (int i = 0; i < n; i++)
            for (int c = 0; c < ncv; c++)
                pmat[c*n + i] = v[vix[i] + SOA_SZ*c];
    else
        for (int i = 0; i < n; i++)
            for (int r = 0; r < nrv; r++)
                for (int c = 0; c < ncv; c++)
                    pmat[(r*ncv + c)*n + i] = v[vix[i] + vrstri[i]*r +
                                                SOA_SZ*c];
}
