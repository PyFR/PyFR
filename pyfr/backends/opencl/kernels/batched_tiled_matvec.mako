<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

inline void
batched_tiled_matvec_accum(__global const ${pcdtype}* tile, const fpdtype_t* xf,
                           int col, int width, fpdtype_t* acc)
{
% if pcdtype == 'half':
    int nvec = width - width % 8, lc = 0;
    for (; lc < nvec; lc += 8)
    {
        float8 v = vloada_half8(0, tile + lc);
        *acc += ${pyfr.dot('v.s{i}', 'xf[col + lc + {i}]', i=8)};
    }
    for (; lc < width; lc++)
        *acc += vload_half(lc, tile)*xf[col + lc];
% elif pcdtype == 'bf16':
    int nvec = width - width % 8, lc = 0;
    for (; lc < nvec; lc += 8)
    {
        float8 v = as_float8(convert_uint8(vload8(0, tile + lc)) << 16);
        *acc += ${pyfr.dot('v.s{i}', 'xf[col + lc + {i}]', i=8)};
    }
    for (; lc < width; lc++)
        *acc += as_float(((uint) tile[lc]) << 16)*xf[col + lc];
% elif pcdtype == 'float':
    int nvec = width - width % 4, lc = 0;
    for (; lc < nvec; lc += 4)
    {
        float4 v = vload4(0, tile + lc);
        *acc += ${pyfr.dot('v.s{i}', 'xf[col + lc + {i}]', i=4)};
    }
    for (; lc < width; lc++)
        *acc += tile[lc]*xf[col + lc];
% else:
    int nvec = width - width % 2, lc = 0;
    for (; lc < nvec; lc += 2)
    {
        double2 v = vload2(0, tile + lc);
        *acc += ${pyfr.dot('v.s{i}', 'xf[col + lc + {i}]', i=2)};
    }
    for (; lc < width; lc++)
        *acc += tile[lc]*xf[col + lc];
% endif
}

__kernel __attribute__((reqd_work_group_size(${mvthreads}, 1, 1)))
void
batched_tiled_matvec(__global const ${pcdtype}* restrict minv,
                     __global const fpdtype_t* restrict x, ixdtype_t ldx,
                     __global fpdtype_t* restrict y, ixdtype_t ldy)
{
<%include file='pyfr.backends.base.kernels.batched_tiled_matvec_body'/>
}
