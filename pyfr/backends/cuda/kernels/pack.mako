# -*- coding: utf-8 -*-
<%inherit file='base'/>

__global__ void
pack_view(int n, int nrv, int ncv,
          const fpdtype_t* __restrict__ v,
          const int* __restrict__ vix,
          const int* __restrict__ vrstri,
          fpdtype_t* __restrict__ pmat)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

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
