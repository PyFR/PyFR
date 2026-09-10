<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:gpukernel name='lu_inv' bounds='${nthreads}'
                 A='out ${ftype}' C='out ${ftype}'
                 inv_out='out ${otype}' src='in ${stype}' eidxs='in ixdtype_t'
                 src_ldim='scalar ixdtype_t' eoff='scalar ixdtype_t'
                 gamma='scalar ${ftype}'>
<%
    FB = fblk
    SB = sblk
    FBP = FB + 1
    SBP = SB + 1
    NT = nthreads
    NBLK = -(-n // SB)
    N2 = n*n
    eye = lambda row, col: f'(({row}) == ({col}) ? ({ftype}) 1 : ({ftype}) 0)'
    src_at = lambda row, col: f'src[{row}*src_ldim + SOA_IX(eoff + bid, {col}, {n})]'
%>
    ixdtype_t bid = PYFR_BLOCK_ID;
    int tid = PYFR_THREAD_ID;
    PYFR_GMEM ${ftype}* a = A + (size_t) bid*${N2};
    PYFR_GMEM ${ftype}* rhs = C + (size_t) bid*${n*sc};
    ixdtype_t oe = eidxs[eoff + bid];

    PYFR_SHARED short ipiv[${n}];
    PYFR_SHARED short piv;
    ${self.argmax_storage(ftype)}
    PYFR_SHARED ${ftype} acc[${max(SB*sc, 3*FB*FBP)}];
    PYFR_SHARED ${ftype} xs[${SB*sc}];
    PYFR_SHARED ${ftype} md[${SB*SBP}];

    // Factor blocks reuse the solve arena; padded ld avoids bank conflicts
    PYFR_LMEM ${ftype}* dblk = acc;
    PYFR_LMEM ${ftype}* Lp = acc + ${FB*FBP};
    PYFR_LMEM ${ftype}* Up = acc + ${2*FB*FBP};

    // Materialise gamma*I - src into a, converting AoSoA source to dense
    for (int i = tid; i < ${N2}; i += ${NT})
    {
        short r = i / ${n}, c = i % ${n};
        ${ftype} v = ${pyfr.fpcast(src_at('r', 'c'), stype, ftype)};
        a[i] = ${eye('r', 'c')} - gamma*v;
    }
    PYFR_SYNC_THREADS();

    // Blocked right-looking LU factorisation with partial pivoting
    for (short pb = 0; pb < ${n}; pb += ${FB})
    {
        short pw = min(${FB}, ${n} - pb);

        // Factor the panel one column at a time with partial pivoting
        for (short j = 0; j < pw; j++)
        {
            // Find the pivot: largest magnitude in column gk on/below the diagonal
            short gk = pb + j;
            ${ftype} mx = 0; short mi = gk;
            for (short r = gk + tid; r < ${n}; r += ${NT})
            {
                ${ftype} v = fabs(a[${n}*r + gk]);
                if (v > mx)
                {
                    mx = v; mi = r;
                }
            }
            ${self.argmax_reduce(ftype, 'mx', 'mi', 'piv')}
            // Record the pivot, then swap the pivot row to the diagonal
            short p = piv;
            if (tid == 0) ipiv[gk] = p + 1;
            if (p != gk)
            {
                for (short c = tid; c < ${n}; c += ${NT})
                {
                    ${ftype} t = a[${n}*gk + c];
                    a[${n}*gk + c] = a[${n}*p + c];
                    a[${n}*p + c] = t;
                }
            }
            PYFR_SYNC_THREADS();
            // Scale the pivot row and eliminate the rows below within the panel
            ${ftype} ipv = (${ftype}) 1 / a[${n}*gk + gk];
            for (short c = gk + 1 + tid; c < pb + pw; c += ${NT})
                a[${n}*gk + c] *= ipv;
            PYFR_SYNC_THREADS();
            for (short r = gk + 1 + tid; r < ${n}; r += ${NT})
            {
                ${ftype} lv = a[${n}*r + gk];
                for (short c = gk + 1; c < pb + pw; c++)
                    a[${n}*r + c] -= lv * a[${n}*gk + c];
            }
            PYFR_SYNC_THREADS();
        }

        if (pb + pw >= ${n}) break;

        // Triangular solve: apply the L panel to the rows to its right
        for (int i = tid; i < pw*pw; i += ${NT})
        {
            short r = i / pw, c = i % pw;
            dblk[${FBP}*r + c] = a[${n}*(pb + r) + pb + c];
        }
        PYFR_SYNC_THREADS();
        for (short c = pb + pw + tid; c < ${n}; c += ${NT})
            for (short r = 0; r < pw; r++)
            {
                ${ftype} s = a[${n}*(pb + r) + c];
                for (short l = 0; l < r; l++)
                    s -= dblk[${FBP}*r + l]*a[${n}*(pb + l) + c];
                a[${n}*(pb + r) + c] = s / dblk[${FBP}*r + r];
            }
        PYFR_SYNC_THREADS();

        // Schur update: subtract L*U from the trailing submatrix, block-tiled
        for (short rb = pb + pw; rb < ${n}; rb += ${FB})
        {
            short br = min(${FB}, ${n} - rb);
            // Lp staged once per rb; the barrier after each Up staging below
            // makes it visible before the GEMM (Lp/Up are disjoint arenas)
            for (int i = tid; i < br*pw; i += ${NT})
            {
                short r = i / pw, l = i % pw;
                Lp[${FBP}*r + l] = a[${n}*(rb + r) + pb + l];
            }
            for (short cb = pb + pw; cb < ${n}; cb += ${FB})
            {
                short bc = min(${FB}, ${n} - cb);
                for (int i = tid; i < pw*bc; i += ${NT})
                {
                    short l = i / bc, c = i % bc;
                    Up[${FBP}*l + c] = a[${n}*(pb + l) + cb + c];
                }
                PYFR_SYNC_THREADS();
% if ftm == 1 and ftn == 1:
                for (int i = tid; i < br*bc; i += ${NT})
                {
                    short r = i / bc, c = i % bc;
                    ${ftype} s = 0;
                    #pragma unroll
                    for (short l = 0; l < ${FB}; l++)
                        s += (l < pw) ? Lp[${FBP}*r + l]*Up[${FBP}*l + c] : 0;
                    a[${n}*(rb + r) + cb + c] -= s;
                }
% else:
                // Register-blocked Schur update: ftm x ftn tile per thread
                int ntr = (br + ${ftm} - 1) / ${ftm};
                int ntc = (bc + ${ftn} - 1) / ${ftn};
                for (int t = tid; t < ntr*ntc; t += ${NT})
                {
                    int r0 = ${ftm}*(t / ntc), c0 = ${ftn}*(t % ntc);
                    ${ftype} cr[${ftm}][${ftn}];
                    #pragma unroll
                    for (int rr = 0; rr < ${ftm}; rr++)
                        #pragma unroll
                        for (int cc = 0; cc < ${ftn}; cc++) cr[rr][cc] = 0;
                    #pragma unroll
                    for (int l = 0; l < ${FB}; l++)
                    {
                        ${ftype} lr[${ftm}], ur[${ftn}];
                        #pragma unroll
                        for (int rr = 0; rr < ${ftm}; rr++)
                            lr[rr] = (l < pw) ? Lp[${FBP}*(r0 + rr) + l] : 0;
                        #pragma unroll
                        for (int cc = 0; cc < ${ftn}; cc++)
                            ur[cc] = (l < pw) ? Up[${FBP}*l + c0 + cc] : 0;
                        #pragma unroll
                        for (int rr = 0; rr < ${ftm}; rr++)
                            #pragma unroll
                            for (int cc = 0; cc < ${ftn}; cc++)
                                cr[rr][cc] += lr[rr]*ur[cc];
                    }
                    #pragma unroll
                    for (int rr = 0; rr < ${ftm}; rr++)
                        #pragma unroll
                        for (int cc = 0; cc < ${ftn}; cc++)
                        {
                            int r = r0 + rr, c = c0 + cc;
                            if (r < br && c < bc)
                                a[${n}*(rb + r) + cb + c] -= cr[rr][cc];
                        }
                }
% endif
                PYFR_SYNC_THREADS();
            }
        }
    }

    // Solve LU X = I one column chunk at a time, writing tiled output
    for (short cc = 0; cc < ${n}; cc += ${sc})
    {
        short ncc = min(${sc}, ${n} - cc);

        for (int i = tid; i < ${n}*ncc; i += ${NT})
        {
            short r = i / ncc, c = i % ncc;
            rhs[${sc}*r + c] = ${eye('r', 'cc + c')};
        }
        PYFR_SYNC_THREADS();

        // Forward blocked lower solve, top-down
        short ib0 = cc / ${SB};
        for (short ib = ib0; ib < ${NBLK}; ib++)
        {
            short row0 = ${SB}*ib, pw = min(${SB}, ${n} - row0);
            for (int i = tid; i < pw*ncc; i += ${NT})
            {
                short r = i / ncc, c = i % ncc;
                acc[${sc}*r + c] = rhs[${sc}*(row0 + r) + c];
            }
            PYFR_SYNC_THREADS();
            for (short jb = ib0; jb < ib; jb++)
            {
                short jrow0 = ${SB}*jb, jw = min(${SB}, ${n} - jrow0);
                for (int i = tid; i < ${SB}*ncc; i += ${NT})
                {
                    short l = i / ncc, c = i % ncc;
                    if (l < jw)
                        xs[${sc}*l + c] = rhs[${sc}*(jrow0 + l) + c];
                    else
                        xs[${sc}*l + c] = 0;
                }
                for (int i = tid; i < ${SB}*pw; i += ${NT})
                {
                    short r = i / ${SB}, l = i % ${SB};
                    if (l < jw)
                        md[${SBP}*r + l] = a[${n}*(row0 + r) + jrow0 + l];
                    else
                        md[${SBP}*r + l] = 0;
                }
                PYFR_SYNC_THREADS();
% if stm == 1 and stn == 1:
                for (int i = tid; i < pw*ncc; i += ${NT})
                {
                    short r = i / ncc, c = i % ncc;
                    ${ftype} s = 0;
                    #pragma unroll
                    for (short l = 0; l < ${SB}; l++)
                        s += md[${SBP}*r + l]*xs[${sc}*l + c];
                    acc[${sc}*r + c] -= s;
                }
% else:
                // Register-blocked GEMM subtract: stm x stn tile per thread
                int ntr = (pw + ${stm} - 1) / ${stm};
                int ntc = (ncc + ${stn} - 1) / ${stn};
                for (int t = tid; t < ntr*ntc; t += ${NT})
                {
                    int r0 = ${stm}*(t / ntc), c0 = ${stn}*(t % ntc);
                    ${ftype} cr[${stm}][${stn}];
                    #pragma unroll
                    for (int rr = 0; rr < ${stm}; rr++)
                        #pragma unroll
                        for (int ccol = 0; ccol < ${stn}; ccol++)
                            cr[rr][ccol] = 0;
                    #pragma unroll
                    for (int l = 0; l < ${SB}; l++)
                    {
                        ${ftype} mr[${stm}], xr[${stn}];
                        #pragma unroll
                        for (int rr = 0; rr < ${stm}; rr++)
                            mr[rr] = md[${SBP}*(r0 + rr) + l];
                        #pragma unroll
                        for (int ccol = 0; ccol < ${stn}; ccol++)
                            xr[ccol] = xs[${sc}*l + c0 + ccol];
                        #pragma unroll
                        for (int rr = 0; rr < ${stm}; rr++)
                            #pragma unroll
                            for (int ccol = 0; ccol < ${stn}; ccol++)
                                cr[rr][ccol] += mr[rr]*xr[ccol];
                    }
                    #pragma unroll
                    for (int rr = 0; rr < ${stm}; rr++)
                        #pragma unroll
                        for (int ccol = 0; ccol < ${stn}; ccol++)
                        {
                            int r = r0 + rr, c = c0 + ccol;
                            if (r < pw && c < ncc)
                                acc[${sc}*r + c] -= cr[rr][ccol];
                        }
                }
% endif
                PYFR_SYNC_THREADS();
            }
            for (int i = tid; i < pw*pw; i += ${NT})
            {
                short k = i / pw, l = i % pw;
                md[${SBP}*k + l] = a[${n}*(row0 + k) + row0 + l];
            }
            PYFR_SYNC_THREADS();
            // Triangular solve against the diagonal block
            for (short c = tid; c < ncc; c += ${NT})
                for (short k = 0; k < pw; k++)
                {
                    ${ftype} s = acc[${sc}*k + c];
                    for (short l = 0; l < k; l++)
                        s -= md[${SBP}*k + l]*acc[${sc}*l + c];
                    acc[${sc}*k + c] = s / md[${SBP}*k + k];
                }
            PYFR_SYNC_THREADS();
            for (int i = tid; i < pw*ncc; i += ${NT})
            {
                short r = i / ncc, c = i % ncc;
                rhs[${sc}*(row0 + r) + c] = acc[${sc}*r + c];
            }
            PYFR_SYNC_THREADS();
        }

        // Backward blocked upper solve, bottom-up
        for (short ib = ${NBLK} - 1; ib >= 0; ib--)
        {
            short row0 = ${SB}*ib, pw = min(${SB}, ${n} - row0);
            for (int i = tid; i < pw*ncc; i += ${NT})
            {
                short r = i / ncc, c = i % ncc;
                acc[${sc}*r + c] = rhs[${sc}*(row0 + r) + c];
            }
            PYFR_SYNC_THREADS();
            for (short jb = ib + 1; jb < ${NBLK}; jb++)
            {
                short jrow0 = ${SB}*jb, jw = min(${SB}, ${n} - jrow0);
                for (int i = tid; i < ${SB}*ncc; i += ${NT})
                {
                    short l = i / ncc, c = i % ncc;
                    if (l < jw)
                        xs[${sc}*l + c] = rhs[${sc}*(jrow0 + l) + c];
                    else
                        xs[${sc}*l + c] = 0;
                }
                for (int i = tid; i < ${SB}*pw; i += ${NT})
                {
                    short r = i / ${SB}, l = i % ${SB};
                    if (l < jw)
                        md[${SBP}*r + l] = a[${n}*(row0 + r) + jrow0 + l];
                    else
                        md[${SBP}*r + l] = 0;
                }
                PYFR_SYNC_THREADS();
% if stm == 1 and stn == 1:
                for (int i = tid; i < pw*ncc; i += ${NT})
                {
                    short r = i / ncc, c = i % ncc;
                    ${ftype} s = 0;
                    #pragma unroll
                    for (short l = 0; l < ${SB}; l++)
                        s += md[${SBP}*r + l]*xs[${sc}*l + c];
                    acc[${sc}*r + c] -= s;
                }
% else:
                // Register-blocked GEMM subtract: stm x stn tile per thread
                int ntr = (pw + ${stm} - 1) / ${stm};
                int ntc = (ncc + ${stn} - 1) / ${stn};
                for (int t = tid; t < ntr*ntc; t += ${NT})
                {
                    int r0 = ${stm}*(t / ntc), c0 = ${stn}*(t % ntc);
                    ${ftype} cr[${stm}][${stn}];
                    #pragma unroll
                    for (int rr = 0; rr < ${stm}; rr++)
                        #pragma unroll
                        for (int ccol = 0; ccol < ${stn}; ccol++)
                            cr[rr][ccol] = 0;
                    #pragma unroll
                    for (int l = 0; l < ${SB}; l++)
                    {
                        ${ftype} mr[${stm}], xr[${stn}];
                        #pragma unroll
                        for (int rr = 0; rr < ${stm}; rr++)
                            mr[rr] = md[${SBP}*(r0 + rr) + l];
                        #pragma unroll
                        for (int ccol = 0; ccol < ${stn}; ccol++)
                            xr[ccol] = xs[${sc}*l + c0 + ccol];
                        #pragma unroll
                        for (int rr = 0; rr < ${stm}; rr++)
                            #pragma unroll
                            for (int ccol = 0; ccol < ${stn}; ccol++)
                                cr[rr][ccol] += mr[rr]*xr[ccol];
                    }
                    #pragma unroll
                    for (int rr = 0; rr < ${stm}; rr++)
                        #pragma unroll
                        for (int ccol = 0; ccol < ${stn}; ccol++)
                        {
                            int r = r0 + rr, c = c0 + ccol;
                            if (r < pw && c < ncc)
                                acc[${sc}*r + c] -= cr[rr][ccol];
                        }
                }
% endif
                PYFR_SYNC_THREADS();
            }
            // Back-substitute against the diagonal block
            for (short c = tid; c < ncc; c += ${NT})
                for (short k = pw - 1; k >= 0; k--)
                {
                    ${ftype} s = acc[${sc}*k + c];
                    for (short l = k + 1; l < pw; l++)
                        s -= a[${n}*(row0 + k) + row0 + l]*acc[${sc}*l + c];
                    acc[${sc}*k + c] = s;
                }
            PYFR_SYNC_THREADS();
            for (int i = tid; i < pw*ncc; i += ${NT})
            {
                short r = i / ncc, c = i % ncc;
                rhs[${sc}*(row0 + r) + c] = acc[${sc}*r + c];
            }
            PYFR_SYNC_THREADS();
        }

        // Write this chunk's inverse columns to tiled output in pivoted order
        for (short c = tid; c < ncc; c += ${NT})
        {
            short gcol = cc + c;
            for (short k = ${n} - 1; k >= 0; k--)
            {
                short pk = ipiv[k] - 1;
                if (gcol == k) gcol = pk;
                else if (gcol == pk) gcol = k;
            }
            for (short r = 0; r < ${n}; r++)
                inv_out[${pyfr.tiled_idx('oe', 'r', 'gcol', trows, tcols, padr, padc)}] = ${pyfr.fpcast(f'rhs[{sc}*r + c]', ftype, otype)};
        }
        PYFR_SYNC_THREADS();
    }
</%pyfr:gpukernel>
