<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%
    ntiles_r = -(-block_size // trows)
    tile_stride = trows*tcols
    row_stride = ntiles_c*tile_stride
    elem_stride = ntiles_r*row_stride
%>
    const int tid = PYFR_THREAD_ID;
    const ixdtype_t midx = PYFR_BLOCK_ID;

% if in_scale:
    const fpdtype_t _in[] = ${pyfr.carray(in_scale)};
% endif
% if out_scale:
    const fpdtype_t _out[] = ${pyfr.carray(out_scale)};
% endif

    PYFR_SHARED fpdtype_t xf[${block_size}];

    for (int idx = tid; idx < ${block_size}; idx += ${mvthreads})
    {
        int upt = idx / ${nvars}, var = idx % ${nvars};
        fpdtype_t v = x[ldx*upt + SOA_IX(midx, var, ${nvars})];
        xf[idx] = ${'_in[var]*v' if in_scale else 'v'};
    }
    PYFR_SYNC_THREADS();

    PYFR_GMEM const ${pcdtype}* minv_e = minv + midx*${elem_stride};

    for (int row = tid; row < ${block_size}; row += ${mvthreads})
    {
        int upt = row / ${nvars}, var = row % ${nvars};
        int tr = row / ${trows}, lr = row % ${trows};

        fpdtype_t acc = 0;
        PYFR_GMEM const ${pcdtype}* tile = minv_e + tr*${row_stride} + lr*${tcols};

        for (int col = 0; col < ${block_size}; col += ${tcols}, tile += ${tile_stride})
        {
            int width = min(${tcols}, ${block_size} - col);

            batched_tiled_matvec_accum(tile, xf, col, width, &acc);
        }

        y[ldy*upt + SOA_IX(midx, var, ${nvars})] = ${'_out[var]*acc' if out_scale else 'acc'};
    }
