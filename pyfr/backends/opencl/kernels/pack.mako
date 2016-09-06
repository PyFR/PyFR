# -*- coding: utf-8 -*-
<%inherit file='base'/>

__kernel void
pack_view(int n, int nrv, int ncv,
          __global const fpdtype_t* restrict v,
          __global const int* restrict vix,
          __global const int* restrict vrstri,
          __global fpdtype_t* restrict pmat)
{
    int i = get_global_id(0);

    if (i < n && ncv == 1)
        pmat[i] = v[vix[i]];
    else if (i < n && nrv == 1)
        for (int c = 0; c < ncv; ++c)
            pmat[c*n + i] = v[vix[i] + SOA_SZ*c];
    else if (i < n)
        for (int r = 0; r < nrv; ++r)
            for (int c = 0; c < ncv; ++c)
                pmat[(r*ncv + c)*n + i] = v[vix[i] + vrstri[i]*r + SOA_SZ*c];
}
