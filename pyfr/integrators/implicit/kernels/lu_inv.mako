<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%
    B, BP, SG, NP = blk, blk + 1, 4, npad
    NR = -(-NP // nthreads)

    # Have lanes which share an item stage the blocks they work on
    stage = nthreads > 1

    # Index of an entry of the factor and of the first right-hand side
    AI = lambda r, c: f'wo + {NP}*({r}) + {c}'
    RA = f'wo + {NP*NP}'

    # Rows and columns trailing the pivot block
    nt = f'{NP} - pb - {B}'

    wout = lambda v: pyfr.fpcast(v, ftype, otype)
    imin = lambda a, b: f'(({a}) < ({b}) ? ({a}) : ({b}))'

    # Count the block rows of the factor and of a run of the solves
    NPB, nrun = NP // B, frb // B

    # Find the last run of the solves and count its block rows
    ibl = (NPB - 1) // nrun * nrun
    nlast = NPB - ibl
%>

## Run stmt for each of the entries i below bound this lane takes, at most nl
<%def name='lane_share(nl, bound, stmt)'>
    for (int q = 0; q < ${nl}; q++)
    {
        int i = p + q*@lsz;
        if (i < ${bound})
            ${stmt};
    }
</%def>

## Load this lane's share of the B-wide slabs of M and X at lb into pm and px
<%def name='fetch_slabs(mo, xo, lb, rh, nml, nxl)'>
<%
    me = f'ws[({mo}) + {NP}*(i / {B}) + ({lb}) + i % {B}]'
    xe = f'ws[({xo}) + {sc}*(({lb}) + i / {sc}) + i % {sc}]'
%>
    ${lane_share(nml, f'({rh})*{B}', f'pm[q] = {me}')}
    ${lane_share(nxl, B*sc, f'px[q] = {xe}')}
</%def>

## Run the body for each of the tm x tn register tiles this lane takes
<%def name='lane_tiles(ntl, ntiles, tm, tn, ntc)'>
    for (int j = 0; j < ${ntl}; j++)
    {
        int t = p + j*@lsz;
        if (t < ${ntiles})
        {
            int r0 = ${tm}*(t / ${ntc}), c0 = ${tn}*(t % ${ntc});
            ${caller.body()}
        }
    }
</%def>

## Run stmt for each entry (rr, cc) of a register tile
<%def name='tile_each(stmt)'>
    for (int rr = 0; rr < ${tm}; rr++)
    {
        for (int cc = 0; cc < ${tn}; cc++)
            ${stmt};
    }
</%def>

<%def name='tile_axpy(tm, tn, op)'>
    // Add or subtract the product of the staged blocks md and xs to the tile
    for (int l = 0; l < ${B}; l++)
    {
        for (int rr = 0; rr < ${tm}; rr++)
        {
            ${ftype} mr = md[${BP}*(r0 + rr) + l];
            for (int cc = 0; cc < ${tn}; cc++)
                cr[j][rr][cc] ${op} mr*xs[${sc}*l + c0 + cc];
        }
    }
</%def>

## Subtract M[rh x k] . X[k x sc] from C[rh x sc], block row so of it into xs
<%def name='gemm_sub_staged(co, mo, xo, k, rh, so, rmax)'>
<%
    ntc = sc // tn
    ntiles = f'(({rh}) / {tm})*{ntc}'
    ntl = -(-(rmax // tm)*ntc // nthreads)
    nml = -(-rmax*B // nthreads)
    nxl = -(-B*sc // nthreads)

    # Number the block row of row r among those the product spans
    brow = lambda r: f'({r}) / {B} % {rmax // B}'
%>
    {
        int p = @lidx;
        ${ftype} cr[${ntl}][${tm}][${tn}] = {{{0}}};
        ${ftype} pm[${nml}], px[${nxl}];

        // Fetch the first slabs of M and X into registers
        if ((${k}) > 0)
        {
            ${fetch_slabs(mo, xo, '0', rh, nml, nxl)}
        }
        for (int lb = 0; lb < ${k}; lb += ${B})
        {
            // Stage the fetched slabs
            ${lane_share(nml, f'({rh})*{B}',
                         f'md[{BP}*(i / {B}) + i % {B}] = pm[q]')}
            ${lane_share(nxl, B*sc, 'xs[i] = px[q]')}
            @sync();

            // Fetch the next slabs while the current ones are consumed
            if (lb + ${B} < ${k})
            {
                ${fetch_slabs(mo, xo, f'lb + {B}', rh, nml, nxl)}
            }

            <%call expr='lane_tiles(ntl, ntiles, tm, tn, ntc)'>
                ${tile_axpy(tm, tn, '+=')}
            </%call>
            @sync();
        }

        // Leave the block row to solve in xs and update the others in place
        <%call expr='lane_tiles(ntl, ntiles, tm, tn, ntc)'>
            int ci = (${co}) + ${sc}*r0 + c0, xi = ${sc}*(r0 - (${so})) + c0;

            if (${brow('r0')} == ${brow(so)})
            {
                ${tile_each(f'xs[xi + {sc}*rr + cc] = '
                            f'ws[ci + {sc}*rr + cc] - cr[j][rr][cc]')}
            }
            else if ((${k}) > 0)
            {
                ${tile_each(f'ws[ci + {sc}*rr + cc] -= cr[j][rr][cc]')}
            }
        </%call>
    }
</%def>

## Solve the B x nc block of arr at off (row stride ld) against the block md
<%def name='diag_solve(lower, arr, off, ld, nc)'>
<%
    grange = range(0, B, SG) if lower else range(B - SG, -1, -SG)
    irange = range(SG) if lower else range(SG - 1, -1, -1)
%>
    {
        int p = @lidx;
% for g in grange:
        #pragma pyfr simd
        for (int c = p; c < ${nc}; c += @lsz)
        {
            ${ftype} s[${SG}];
%   for i in range(SG):
            s[${i}] = ${arr}[(${off}) + (${ld})*${g + i} + c];
%   endfor

            // Eliminate the rows outside this group of rows
%   for l in (range(g) if lower else range(g + SG, B)):
            {
                ${ftype} x = ${arr}[(${off}) + (${ld})*${l} + c];
%     for i in range(SG):
                s[${i}] -= md[${BP*(g + i) + l}]*x;
%     endfor
            }
%   endfor

            // Solve the triangle within the group
%   for i in irange:
%     for ii in (range(i) if lower else range(i + 1, SG)):
            s[${i}] -= md[${BP*(g + i) + g + ii}]*s[${ii}];
%     endfor
%     if lower:
            s[${i}] /= md[${BP*(g + i) + g + i}];
%     endif
            ${arr}[(${off}) + (${ld})*${g + i} + c] = s[${i}];
%   endfor
        }
% endfor
    }
    @sync();
</%def>

## Copy the column chunk of the U panel at cb into xs, or back out of it
<%def name='copy_chunk(back)'>
<% xe, ae = f'xs[{sc}*r + c]', f"ws[{AI('pb + r', 'cb + c')}]" %>
    for (int i = p; i < ${B}*cw; i += @lsz)
    {
        int r = i / cw, c = i % cw;
        ${f'{ae} = {xe}' if back else f'{xe} = {ae}'};
    }
    @sync();
</%def>

## Solve block row ib of a triangular system given the rows already solved
<%def name='tri_block(lower, mo, xo, k)'>
<%
    gargs = ('ws', 'ra', sc, 'ws', mo, NP, 'ws', f'{RA} + {xo}', sc, k, tm, tn)
%>
    int row0 = ${B}*ib, ra = ${RA} + ${sc}*row0;

    // Subtract the product of the factor with the rows already solved
    ${gemm_sub_direct(*gargs, B // tm, sc // tn)}
    ${stage_diag('row0', 'row0')}

    // Solve against the diagonal block, lanes over columns
    ${diag_solve(lower, 'ws', 'ra', sc, sc)}
</%def>

## Solve the block row at row0 given k rows from mc, updating rh rows from c0
<%def name='tri_staged(lower, c0, rh, mc, k, rmax)'>
<% co, mo, xo = f'{RA} + {sc}*({c0})', AI(c0, mc), f'{RA} + {sc}*({mc})' %>
    int ra = ${RA} + ${sc}*row0;

    // Subtract the product of the factor with the rows already solved
    ${gemm_sub_staged(co, mo, xo, k, rh, f'row0 - ({c0})', rmax)}
    ${stage_diag('row0', 'row0')}

    // Solve against the diagonal block, lanes over columns
    ${diag_solve(lower, 'xs', '0', sc, sc)}

    // Write the solved block back
    for (int i = @lidx; i < ${B*sc}; i += @lsz)
        ws[ra + i] = xs[i];
    @sync();
</%def>

## Subtract M . X from C with the operands read in place over ntr x ntc tiles
<%def name='gemm_sub_direct(C, co, ldc, M, mo, ldm, X, xo, ldx, k, tm, tn,
                            ntr, ntc)'>
    for (int t = @lidx; t < (${ntr})*(${ntc}); t += @lsz)
    {
        int r0 = ${tm}*(t / (${ntc}));
        int c0 = ${tn}*(t % (${ntc}));
        ${ftype} cr[${tm}][${tn}] = {{0}};

        // Accumulate the product a block of rows at a time
        for (int lb = 0; lb < ${k}; lb += ${B})
        {
% for l in range(B):
            {
                int mi = (${mo}) + (${ldm})*r0 + lb + ${l};
                int xi = (${xo}) + (${ldx})*(lb + ${l}) + c0;
                for (int rr = 0; rr < ${tm}; rr++)
                {
                    ${ftype} mr = ${M}[mi + (${ldm})*rr];
                    #pragma pyfr simd
                    for (int cc = 0; cc < ${tn}; cc++)
                        cr[rr][cc] += mr*${X}[xi + cc];
                }
            }
% endfor
        }
        for (int rr = 0; rr < ${tm}; rr++)
        {
            int ci = (${co}) + (${ldc})*(r0 + rr) + c0;
            #pragma pyfr simd
            for (int cc = 0; cc < ${tn}; cc++)
                ${C}[ci + cc] -= cr[rr][cc];
        }
    }
</%def>

## Load this lane's share of the L blocks of the frb rows at rb into pm
<%def name='fetch_lrows(rb, nrl)'>
<% rows = imin(frb, f'{NP} - ({rb})') %>
    ${lane_share(nrl, f'{rows}*{B}',
                 f"pm[q] = ws[{AI(f'({rb}) + i / {B}', f'pb + i % {B}')}]")}
</%def>

## Subtract L . U from the trailing matrix with U staged once per chunk
<%def name='schur_staged()'>
<%
    ntiles = (frb // tm)*(sc // tn)
    ntl = -(-ntiles // nthreads)
    nrl = -(-frb*B // nthreads)
%>
    for (int cb = pb + ${B}; cb < ${NP}; cb += ${sc})
    {
        int p = @lidx, cw = ${imin(sc, f'{NP} - cb')}, ntc = cw / ${tn};

        // Stage this column chunk of the U panel
        ${copy_chunk(back=False)}

        // Fetch the L blocks of the first row pass into registers
        ${ftype} pm[${nrl}];
        ${fetch_lrows(f'pb + {B}', nrl)}
        for (int rb = pb + ${B}; rb < ${NP}; rb += ${frb})
        {
            int rh = ${imin(frb, f'{NP} - rb')}, ntr = rh / ${tm};

            // Load the C tiles of this pass ahead of the staging barrier
            ${ftype} cr[${ntl}][${tm}][${tn}];
            <%call expr='lane_tiles(ntl, "ntr*ntc", tm, tn, "ntc")'>
                int ci = ${AI('rb + r0', 'cb + c0')};
                ${tile_each(f'cr[j][rr][cc] = ws[ci + {NP}*rr + cc]')}
            </%call>

            // Stage the L blocks of these rows
            ${lane_share(nrl, f'rh*{B}',
                         f'md[{BP}*(i / {B}) + i % {B}] = pm[q]')}
            @sync();

            // Fetch the L blocks of the next row pass while these are used
            if (rb + ${frb} < ${NP})
            {
                ${fetch_lrows(f'rb + {frb}', nrl)}
            }

            <%call expr='lane_tiles(ntl, "ntr*ntc", tm, tn, "ntc")'>
                int ci = ${AI('rb + r0', 'cb + c0')};
                ${tile_axpy(tm, tn, '-=')}

                // Write the tile back
                ${tile_each(f'ws[ci + {NP}*rr + cc] = cr[j][rr][cc]')}
            </%call>
            @sync();
        }
    }
</%def>

<%def name='stage_diag(r0, c0)'>
    // Stage the diagonal block of the factor into the padded array md
    for (int i = @lidx; i < ${B*B}; i += @lsz)
    {
        int r = i / ${B}, c = i % ${B};
        md[${BP}*r + c] = ws[${AI(f'{r0} + r', f'{c0} + c')}];
    }
    @sync();
</%def>

## Load this lane's rows of the panel at pb into pv, or write them back
<%def name='copy_panel(back)'>
<% pe, ae = 'pv[q][c]', f"ws[{AI('r', 'pb + c')}]" %>
    for (int q = 0; q < ${NR}; q++)
    {
        int r = p + q*@lsz;
        if (r >= pb && r < ${NP})
        {
            for (int c = 0; c < ${B}; c++)
                ${f'{ae} = {pe}' if back else f'{pe} = {ae}'};
        }
    }
</%def>

<%pyfr:kernel name='lu_inv' ndim='1' coop='${str(nthreads)}'
              src='in ${stype}[${str(n)}][${str(n)}]'
              inv='out tiled view ${otype}[${str(n)}][${str(n)}]'
              scale='scalar fpdtype_t'
              ws='inout broadcast ${ftype}[${str(nwork*wsz)}]'
              perm='inout broadcast int16_t[${str(nwork*npad)}]'
              wk='in ixdtype_t'>
    // Storage shared by the lanes working on an item
    shared ${ftype} md[${frb*BP}];
% if stage:
    shared ${ftype} xs[${B*sc}];
% endif
    shared ${ftype} pr[${B}], pg[${B}];
    shared int16_t ipiv[${B}];

    // Offsets of the workspace and row order of the item
    ixdtype_t wo = wk*${wsz}, po = wk*${NP};

    // Convert the scale to the factorisation precision
    ${ftype} scl = (${ftype}) scale;

    // Materialise I - scale*src, padded with the identity to NP x NP
    for (int r = 0; r < ${NP}; r++)
    {
        for (int c = @lidx; c < ${NP}; c += @lsz)
        {
            ${ftype} v = 0;
            if (r < ${n} && c < ${n})
                v = ${pyfr.fpcast('src[r][c]', stype, ftype)};
            ws[${AI('r', 'c')}] = (r == c) - scl*v;
        }
    }

    // Start from the rows in their given order
    for (int r = @lidx; r < ${NP}; r += @lsz)
        perm[po + r] = r;
    @sync();

    // Factorise by blocked right-looking LU with partial pivoting
    for (int pb = 0; pb < ${NP}; pb += ${B})
    {
        int p = @lidx;

        // Load this lane's rows of the panel into registers
        ${ftype} pv[${NR}][${B}];
        ${copy_panel(back=False)}

        // Factor the panel one column at a time
        for (int j = 0; j < ${B}; j++)
        {
            int gk = pb + j;

            // Find the pivot: largest magnitude in column j on or below gk
            ${ftype} mx = 0;
            int mi = gk;
            for (int q = 0; q < ${NR}; q++)
            {
                int r = p + q*@lsz;
                if (r >= gk && r < ${NP})
                {
                    ${ftype} v = fabs(pv[q][j]);
                    if (v > mx)
                    {
                        mx = v;
                        mi = r;
                    }
                }
            }
            #pragma pyfr allreduce(mx, mi)
            {
                ${ftype} mx_o = @partner(mx);
                int mi_o = @partner(mi);
                if (mx_o > mx || (mx_o == mx && mi_o < mi))
                {
                    mx = mx_o;
                    mi = mi_o;
                }
            }

            // Record the pivot and stage the pivot and diagonal rows
            if (p == 0)
                ipiv[j] = mi;
% if stage:
            for (int q = 0; q < ${NR}; q++)
            {
                int r = p + q*@lsz;
                if (r == mi)
                {
                    for (int c = 0; c < ${B}; c++)
                        pr[c] = pv[q][c];
                }
                if (r == gk)
                {
                    for (int c = 0; c < ${B}; c++)
                        pg[c] = pv[q][c];
                }
            }
% else:
            for (int c = 0; c < ${B}; c++)
            {
                pr[c] = pv[mi][c];
                pg[c] = pv[gk][c];
            }
% endif
            @sync();

            // Swap in the pivot row, scale it and eliminate the rows below
            ${ftype} ipv = (${ftype}) 1 / pr[j];
            for (int q = 0; q < ${NR}; q++)
            {
                int r = p + q*@lsz;
                if (r == gk)
                {
                    #pragma pyfr simd
                    for (int c = 0; c < ${B}; c++)
                        pv[q][c] = c > j ? pr[c]*ipv : pr[c];
                }
                else if (r > gk && r < ${NP})
                {
                    if (r == mi)
                    {
                        for (int c = 0; c < ${B}; c++)
                            pv[q][c] = pg[c];
                    }
                    ${ftype} lv = pv[q][j];
                    #pragma pyfr simd
                    for (int c = 0; c < ${B}; c++)
                        pv[q][c] -= c > j ? lv*(pr[c]*ipv) : 0;
                }
            }
        }

        // Write the factored panel back
        ${copy_panel(back=True)}

        // Apply the panel's row interchanges to the remaining columns
        for (int c = p; c < ${NP}; c += @lsz)
        {
            if (c >= pb && c < pb + ${B})
                continue;
            for (int j = 0; j < ${B}; j++)
            {
                int gk = pb + j, mi = ipiv[j];
                if (mi != gk)
                {
                    ${ftype} t = ws[${AI('gk', 'c')}];
                    ws[${AI('gk', 'c')}] = ws[${AI('mi', 'c')}];
                    ws[${AI('mi', 'c')}] = t;
                }
            }
        }

        // Apply them to the order of the rows
        if (p == 0)
        {
            for (int j = 0; j < ${B}; j++)
            {
                int gk = pb + j, mi = ipiv[j], rk = perm[po + gk];
                perm[po + gk] = perm[po + mi];
                perm[po + mi] = rk;
            }
        }
        @sync();

        if (pb + ${B} >= ${NP})
            break;

        ${stage_diag('pb', 'pb')}

        // Solve the U panel against the diagonal block
% if stage:
        for (int cb = pb + ${B}; cb < ${NP}; cb += ${sc})
        {
            int p = @lidx, cw = ${imin(sc, f'{NP} - cb')};

            // Stage this column chunk of the U panel
            ${copy_chunk(back=False)}

            // Solve the chunk against the diagonal block
            ${diag_solve(True, 'xs', '0', sc, 'cw')}

            // Write the solved chunk back
            ${copy_chunk(back=True)}
        }
% else:
        ${diag_solve(True, 'ws', AI('pb', f'pb + {B}'), NP, nt)}
% endif

        // Apply the Schur update to the trailing matrix
% if stage:
        ${schur_staged()}
% else:
        ${gemm_sub_direct('ws', AI(f'pb + {B}', f'pb + {B}'), NP,
                          'ws', AI(f'pb + {B}', 'pb'), NP,
                          'ws', AI('pb', f'pb + {B}'), NP, B, tm, tn,
                          f'({nt}) / {tm}', f'({nt}) / {tn}')}
% endif
        @sync();
    }

    // Solve LU X = I one column chunk at a time
    for (int cc = 0; cc < ${NP}; cc += ${sc})
    {
        // Start from this chunk of the columns of the identity
        for (int r = 0; r < ${NP}; r++)
        {
            for (int c = @lidx; c < ${sc}; c += @lsz)
                ws[${RA} + ${sc}*r + c] = r == cc + c;
        }
        @sync();

% if stage:
        // Solve the lower triangle a run of block rows at a time, top-down
        int ib0 = cc / ${B};
        for (int ibr = ib0, ie = 0; ibr < ${NPB}; ibr = ie)
        {
            // Find the block row which ends the run this block row is in
            int ira = ibr / ${nrun} * ${nrun};
            ie = ira + (ira == ${ibl} ? ${nlast} : ${nrun});

            // Solve the first block row, updating the run for the rows above
            {
                int row0 = ${B}*ibr, rh = ${B}*(ie - ibr);

                ${tri_staged(True, 'row0', 'rh', f'{B}*ib0', f'row0 - {B}*ib0',
                             frb)}
            }

            // Solve the other block rows of the run given the ones above them
            for (int ib = ibr + 1; ib < ie; ib++)
            {
                int row0 = ${B}*ib;

                ${tri_staged(True, 'row0', B, f'{B}*ibr', f'row0 - {B}*ibr',
                             B)}
            }
        }

        // Solve the upper triangle a run of block rows at a time, bottom-up
        for (int ie = ${NPB}, ibr = 0; ie > 0; ie = ibr)
        {
            // Find the block row which starts the run the last block row is in
            ibr = (ie - 1) / ${nrun} * ${nrun};

            // Solve the last block row, updating the run for the rows below
            {
                int row0 = ${B}*(ie - 1), rh = ${B}*(ie - ibr);

                ${tri_staged(False, f'{B}*ibr', 'rh', f'row0 + {B}',
                             f'{NP} - row0 - {B}', frb)}
            }

            // Solve the other block rows of the run given the ones below them
            for (int ib = ie - 2; ib >= ibr; ib--)
            {
                int row0 = ${B}*ib;

                ${tri_staged(False, 'row0', B, f'row0 + {B}',
                             f'{B}*ie - row0 - {B}', B)}
            }
        }
% else:
        // Solve the lower triangle a block row at a time, top-down
        int ib0 = cc / ${B};
        for (int ib = ib0; ib < ${NP // B}; ib++)
        {
            ${tri_block(True, AI('row0', f'{B}*ib0'), f'{sc*B}*ib0',
                        f'row0 - {B}*ib0')}
        }

        // Solve the upper triangle a block row at a time, bottom-up
        for (int ib = ${NP // B} - 1; ib >= 0; ib--)
        {
            ${tri_block(False, AI('row0', f'row0 + {B}'),
                        f'{sc}*(row0 + {B})', f'{NP} - row0 - {B}')}
        }
% endif

        // Write this chunk of the inverse out in pivoted order
        for (int c = @lidx; c < ${imin(sc, f'{NP} - cc')}; c += @lsz)
        {
            int gcol = perm[po + cc + c];
            if (gcol < ${n})
            {
                for (int r = 0; r < ${n}; r++)
                    inv[r][gcol] = ${wout(f'ws[{RA} + {sc}*r + c]')};
            }
        }
        @sync();
    }
</%pyfr:kernel>
