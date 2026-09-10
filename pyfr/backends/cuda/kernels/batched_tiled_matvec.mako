<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__device__ inline void
batched_tiled_matvec_accum(const ${pcdtype}* tile, const fpdtype_t* xf,
                           int col, int width, fpdtype_t* acc)
{
% if pcdtype in ('half', 'bf16'):
    int nvec = width - width % 8, lc = 0;
    for (; lc < nvec; lc += 8)
    {
        uint4 v = *((const uint4*)(tile + lc));
        ${pcdtype} h[8];
        h[0].bits = v.x & 0xFFFF; h[1].bits = v.x >> 16;
        h[2].bits = v.y & 0xFFFF; h[3].bits = v.y >> 16;
        h[4].bits = v.z & 0xFFFF; h[5].bits = v.z >> 16;
        h[6].bits = v.w & 0xFFFF; h[7].bits = v.w >> 16;

        *acc += ${pyfr.dot('(fpdtype_t) h[{i}]', 'xf[col + lc + {i}]', i=8)};
    }
    for (; lc < width; lc++)
        *acc += (fpdtype_t)tile[lc]*xf[col + lc];
% elif pcdtype == 'float':
    int nvec = width - width % 4, lc = 0;
    for (; lc < nvec; lc += 4)
    {
        float4 v = *((const float4*)(tile + lc));
        fpdtype_t vv[4] = {v.x, v.y, v.z, v.w};

        *acc += ${pyfr.dot('vv[{i}]', 'xf[col + lc + {i}]', i=4)};
    }
    for (; lc < width; lc++)
        *acc += (fpdtype_t)tile[lc]*xf[col + lc];
% else:
    int nvec = width - width % 2, lc = 0;
    for (; lc < nvec; lc += 2)
    {
        double2 v = *((const double2*)(tile + lc));
        fpdtype_t vv[2] = {v.x, v.y};

        *acc += ${pyfr.dot('vv[{i}]', 'xf[col + lc + {i}]', i=2)};
    }
    for (; lc < width; lc++)
        *acc += tile[lc]*xf[col + lc];
% endif
}

__global__ __launch_bounds__(${mvthreads}, 16) void
batched_tiled_matvec(const ${pcdtype}* __restrict__ minv,
                     const fpdtype_t* __restrict__ x, ixdtype_t ldx,
                     fpdtype_t* __restrict__ y, ixdtype_t ldy)
{
<%include file='pyfr.backends.base.kernels.batched_tiled_matvec_body'/>
}
