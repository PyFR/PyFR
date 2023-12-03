<%inherit file='base'/>

__kernel void
pack_view(ixdtype_t n, ixdtype_t nrv, ixdtype_t ncv,
          __global const fpdtype_t* restrict v,
          __global const int* restrict vix,
          __global const int* restrict vrstri,
          __global fpdtype_t* restrict pmat)
{
    ixdtype_t i = get_global_id(0);

    if (i < n && ncv == 1)
        pmat[i] = v[vix[i]];
    else if (i < n && nrv == 1)
        for (ixdtype_t c = 0; c < ncv; ++c)
            pmat[c*n + i] = v[vix[i] + SOA_SZ*c];
    else if (i < n)
        for (ixdtype_t r = 0; r < nrv; ++r)
            for (ixdtype_t c = 0; c < ncv; ++c)
                pmat[(r*ncv + c)*n + i] = v[vix[i] + vrstri[i]*r + SOA_SZ*c];
}
