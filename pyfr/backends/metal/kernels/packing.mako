<%inherit file='base'/>

kernel void
pack_view(constant ixdtype_t& n, constant ixdtype_t& nrv,
          constant ixdtype_t& ncv,
          device const fpdtype_t* v,
          device const ixdtype_t* vix,
          device const ixdtype_t* vrstri,
          device fpdtype_t* pmat,
          uint i [[thread_position_in_grid]])
{
    if (i < n && ncv == 1)
        pmat[i] = v[vix[i]];
    else if (i < n && nrv == 1)
        for (ixdtype_t c = 0; c < ncv; ++c)
            pmat[c*n + i] = v[vix[i] + SOA_SZ*c];
    else if (i < n)
        for (ixdtype_t r = 0; r < nrv; ++r)
            for (int c = 0; c < ncv; ++c)
                pmat[(r*ncv + c)*n + i] = v[vix[i] + vrstri[i]*r + SOA_SZ*c];
}

kernel void
unpack_view(constant ixdtype_t& n, constant ixdtype_t& nrv,
            constant ixdtype_t& ncv,
            device fpdtype_t* v,
            device const ixdtype_t* vix,
            device const ixdtype_t* vrstri,
            device const fpdtype_t* pmat,
            uint i [[thread_position_in_grid]])
{
    if (i < n && ncv == 1)
        v[vix[i]] = pmat[i];
    else if (i < n && nrv == 1)
        for (ixdtype_t c = 0; c < ncv; ++c)
            v[vix[i] + SOA_SZ*c] = pmat[c*n + i];
    else if (i < n)
        for (ixdtype_t r = 0; r < nrv; ++r)
            for (int c = 0; c < ncv; ++c)
                v[vix[i] + vrstri[i]*r + SOA_SZ*c] = pmat[(r*ncv + c)*n + i];
}
