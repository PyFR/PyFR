<%inherit file='base'/>

kernel void
pack_view(constant int& n, constant int& nrv, constant int& ncv,
          device const fpdtype_t* v,
          device const int* vix,
          device const int* vrstri,
          device fpdtype_t* pmat,
          uint i [[thread_position_in_grid]])
{
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
