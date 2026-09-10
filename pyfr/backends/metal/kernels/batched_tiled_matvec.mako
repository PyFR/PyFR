<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

inline void
batched_tiled_matvec_accum(device const ${pcdtype}* tile,
                           threadgroup const fpdtype_t* xf,
                           int col, int width, thread fpdtype_t* acc)
{
    int nvec = width - width % 4, lc = 0;
    for (; lc < nvec; lc += 4)
    {
% if pcdtype == 'bf16':
        bfloat4 v = *(device const bfloat4*) (tile + lc);
% else:
        ${pcdtype}4 v = *(device const ${pcdtype}4*) (tile + lc);
% endif
        *acc += ${pyfr.dot('(fpdtype_t) v[{i}]', 'xf[col + lc + {i}]', i=4)};
    }
    for (; lc < width; lc++)
        *acc += (fpdtype_t) tile[lc]*xf[col + lc];
}

kernel void
batched_tiled_matvec(device const ${pcdtype}* minv,
                     device const fpdtype_t* x, constant ixdtype_t& ldx,
                     device fpdtype_t* y, constant ixdtype_t& ldy,
                     uint3 tid_ [[thread_position_in_threadgroup]],
                     uint3 bid_ [[threadgroup_position_in_grid]])
{
<%include file='pyfr.backends.base.kernels.batched_tiled_matvec_body'/>
}
