<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%
x_leaddim = nvars*csubsz
blocksz = nupts*x_leaddim
mixed = pcdtype != pyfr.npdtype_to_ctype(fpdtype)
%>

struct batched_tiled_matvec_kargs
{
    ixdtype_t neles;
    ${pcdtype} *minv_v;
    fpdtype_t *x_v, *y_v;
};

void batched_tiled_matvec(const struct batched_tiled_matvec_kargs *restrict args)
{
    ixdtype_t neles = args->neles;
    ${pcdtype} *minv_v = args->minv_v;
    fpdtype_t *x_v = args->x_v, *y_v = args->y_v;

% if in_scale:
    static const fpdtype_t in_scale[] = ${pyfr.carray(in_scale)};
% endif
% if out_scale:
    static const fpdtype_t out_scale[] = ${pyfr.carray(out_scale)};
% endif

    #define ROW_PTR(buf, upt, var) ((buf) + (upt)*${x_leaddim} + (s*${nvars} + (var))*SOA_SZ)
    #define TILE_PTR(tr, tc, lr) (minv + (tr)*${ntiles_c*tcols**2*csubsz} + (tc)*${tcols**2*csubsz} + (lr)*${tcols*csubsz} + s*${tcols*soasz})

    #pragma omp parallel for ${schedule}
    for (ixdtype_t ib = 0; ib < (neles + BLK_SZ - 1) / BLK_SZ; ib++)
    {
        ixdtype_t rem = neles - ib*BLK_SZ;
        ${pcdtype} *minv = minv_v + ib*${ntiles_c**2*tcols**2*csubsz};
        fpdtype_t *x = x_v + ib*${blocksz}, *y = y_v + ib*${blocksz};

        for (ixdtype_t s = 0; s < ${csubsz // soasz}; s++)
        {
            ixdtype_t raw = rem - s*SOA_SZ;
            if (raw <= 0) continue;
            ixdtype_t active = min(raw, (ixdtype_t)SOA_SZ);

% if tcols > 8:
            for (ixdtype_t tr = 0; tr < ${ntiles_c}; tr++)
            {
                ixdtype_t row0 = tr*${tcols};
                ixdtype_t nrowt = min(${tcols}, ${block_size} - row0);
                alignas(64) fpdtype_t acc[${tcols}][SOA_SZ] = {{0}};

                for (ixdtype_t tc = 0; tc < ${ntiles_c}; tc++)
                {
                    ixdtype_t col0 = tc*${tcols};
                    ixdtype_t ncolt = min(${tcols}, ${block_size} - col0);
                    alignas(64) fpdtype_t xcache[${tcols}][SOA_SZ];

                    for (ixdtype_t lc = 0; lc < ncolt; lc++)
                    {
                        ixdtype_t col = col0 + lc;
                        ixdtype_t x_upt = col / ${nvars}, x_var = col % ${nvars};
                        fpdtype_t *x_ptr = ROW_PTR(x, x_upt, x_var);

                        #pragma omp simd
                        for (ixdtype_t lane = 0; lane < active; lane++)
                            xcache[lc][lane] = ${'in_scale[x_var]*' if in_scale else ''}x_ptr[lane];
                    }

                    for (ixdtype_t lr = 0; lr < nrowt; lr++)
                    {
                        ${pcdtype} *trow = TILE_PTR(tr, tc, lr);

% if mixed:
                        alignas(64) fpdtype_t mcache[${tcols}][SOA_SZ];
                        for (ixdtype_t lc = 0; lc < ncolt; lc++)
                        {
                            ${pcdtype} *m_ptr = trow + lc*SOA_SZ;
                            #pragma omp simd
                            for (ixdtype_t lane = 0; lane < SOA_SZ; lane++)
% if pcdtype == 'bf16':
                                mcache[lc][lane] = bf16_to_f32(m_ptr[lane]);
% else:
                                mcache[lc][lane] = m_ptr[lane];
% endif
                        }
% endif

                        for (ixdtype_t lc = 0; lc < ncolt; lc++)
                        {
                            fpdtype_t *mp = ${'mcache[lc]' if mixed else 'trow + lc*SOA_SZ'};

                            #pragma omp simd
                            for (ixdtype_t lane = 0; lane < active; lane++)
                                acc[lr][lane] += mp[lane]*xcache[lc][lane];
                        }
                    }
                }

                for (ixdtype_t row = row0; row < row0 + nrowt; row++)
                {
                    ixdtype_t y_upt = row / ${nvars}, y_var = row % ${nvars};
                    fpdtype_t *y_ptr = ROW_PTR(y, y_upt, y_var);

                    #pragma omp simd
                    for (ixdtype_t lane = 0; lane < active; lane++)
                        y_ptr[lane] = ${'out_scale[y_var]*' if out_scale else ''}acc[row - row0][lane];
                }
            }
% else:
            for (ixdtype_t row = 0; row < ${block_size}; row++)
            {
                alignas(64) fpdtype_t acc[SOA_SZ] = {0};
                ixdtype_t tr = row / ${tcols}, lr = row % ${tcols};

                for (ixdtype_t tc = 0; tc < ${ntiles_c}; tc++)
                {
                    ${pcdtype} *trow = TILE_PTR(tr, tc, lr);
                    ixdtype_t col0 = tc*${tcols};
                    ixdtype_t ncolt = min(${tcols}, ${block_size} - col0);

                    for (ixdtype_t lc = 0; lc < ncolt; lc++)
                    {
                        ixdtype_t col = col0 + lc;
                        ixdtype_t x_upt = col / ${nvars}, x_var = col % ${nvars};
                        ${pcdtype} *m_ptr = trow + lc*SOA_SZ;
                        fpdtype_t *x_ptr = ROW_PTR(x, x_upt, x_var);

                        #pragma omp simd
                        for (ixdtype_t lane = 0; lane < active; lane++)
                            acc[lane] += m_ptr[lane]*${'(in_scale[x_var]*x_ptr[lane])' if in_scale else 'x_ptr[lane]'};
                    }
                }

                ixdtype_t y_upt = row / ${nvars}, y_var = row % ${nvars};
                fpdtype_t *y_ptr = ROW_PTR(y, y_upt, y_var);

                #pragma omp simd
                for (ixdtype_t lane = 0; lane < active; lane++)
                    y_ptr[lane] = ${'out_scale[y_var]*' if out_scale else ''}acc[lane];
            }
% endif
        }
    }

    #undef ROW_PTR
    #undef TILE_PTR
}
