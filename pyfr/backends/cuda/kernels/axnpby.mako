<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
axnpby(ixdtype_t ncolb, ixdtype_t ldim,
       fpdtype_t* __restrict__ x0,
       ${', '.join(f'const fpdtype_t* __restrict__ x{i}'
                   for i in range(1, nv))}${',' if nv > 1 else ''}
       ${', '.join(f'fpdtype_t a{i}' for i in range(nv))})
{
% if in_scale:
    const fpdtype_t _in[] = ${pyfr.carray(in_scale)};
% endif
% if out_scale:
    const fpdtype_t _out[] = ${pyfr.carray(out_scale)};
% endif
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    ixdtype_t j = ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x;
    ixdtype_t idx;

    if (j < ncolb && a0 == 0.0)
    {
    % for k in range(ncola):
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] = ${pyfr.axnpby_expr(k, 'idx', 1, nv=nv, in_scale_idxs=in_scale_idxs, out_scale=out_scale)};
    % endfor
    }
% if nv > 1:
    else if (j < ncolb && a0 == 1.0)
    {
    % for k in range(ncola):
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] += ${pyfr.axnpby_expr(k, 'idx', 1, nv=nv, in_scale_idxs=in_scale_idxs, out_scale=out_scale)};
    % endfor
    }
% endif
    else if (j < ncolb)
    {
    % for k in range(ncola):
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] = ${pyfr.axnpby_expr(k, 'idx', 0, nv=nv, in_scale_idxs=in_scale_idxs, out_scale=out_scale)};
    % endfor
    }
}
