<%def name="pack_view_body()">
    if (i < n && ncv == 1)
        pmat[i] = v[vix[i]];
    else if (i < n && nrv == 1)
        for (ixdtype_t c = 0; c < ncv; ++c)
            pmat[c*n + i] = v[vix[i] + SOA_SZ*c];
    else if (i < n)
        for (ixdtype_t r = 0; r < nrv; ++r)
            for (ixdtype_t c = 0; c < ncv; ++c)
                pmat[(r*ncv + c)*n + i] = v[vix[i] + vrstri[i]*r + SOA_SZ*c];
</%def>

<%def name="unpack_view_body()">
    if (i < n && ncv == 1)
        v[vix[i]] = pmat[i];
    else if (i < n && nrv == 1)
        for (ixdtype_t c = 0; c < ncv; ++c)
            v[vix[i] + SOA_SZ*c] = pmat[c*n + i];
    else if (i < n)
        for (ixdtype_t r = 0; r < nrv; ++r)
            for (ixdtype_t c = 0; c < ncv; ++c)
                v[vix[i] + vrstri[i]*r + SOA_SZ*c] = pmat[(r*ncv + c)*n + i];
</%def>
