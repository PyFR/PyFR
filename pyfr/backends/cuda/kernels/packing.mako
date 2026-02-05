<%inherit file='base'/>

__global__ void
pack_view(ixdtype_t n, ixdtype_t nrv, ixdtype_t ncv,
          const fpdtype_t* __restrict__ v,
          const ixdtype_t* __restrict__ vix,
          const ixdtype_t* __restrict__ vrstri,
          fpdtype_t* __restrict__ pmat)
{
    ixdtype_t i = ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x;

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

__global__ void
unpack_view(ixdtype_t n, ixdtype_t nrv, ixdtype_t ncv,
            fpdtype_t* __restrict__ v,
            const ixdtype_t* __restrict__ vix,
            const ixdtype_t* __restrict__ vrstri,
            const fpdtype_t* __restrict__ pmat)
{
    ixdtype_t i = ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x;

    if (i < n && ncv == 1)
        v[vix[i]] = pmat[i];
    else if (i < n && nrv == 1)
        for (ixdtype_t c = 0; c < ncv; ++c)
            v[vix[i] + SOA_SZ*c] = pmat[c*n + i];
    else if (i < n)
        for (ixdtype_t r = 0; r < nrv; ++r)
            for (ixdtype_t c = 0; c < ncv; ++c)
                v[vix[i] + vrstri[i]*r + SOA_SZ*c] = pmat[(r*ncv + c)*n + i];
}
