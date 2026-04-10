<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void
axnpby(ixdtype_t ncolb, ixdtype_t ldim,
       __global fpdtype_t* restrict x0,
       ${', '.join(f'__global const fpdtype_t* restrict x{i}'
                   for i in range(1, nv))}${',' if nv > 1 else ''}
       ${', '.join(f'fpdtype_t a{i}' for i in range(nv))})
{
% if in_scale:
    const fpdtype_t _in[] = ${pyfr.carray(in_scale)};
% endif
% if out_scale:
    const fpdtype_t _out[] = ${pyfr.carray(out_scale)};
% endif
    ixdtype_t i = get_global_id(1), j = get_global_id(0);
    ixdtype_t idx;

    if (j < ncolb && a0 == 0)
    {
    % for k in range(ncola):
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] = ${pyfr.axnpby_expr(k, 'idx', 1, nv=nv, in_scale_idxs=in_scale_idxs, out_scale=out_scale)};
    % endfor
    }
% if nv > 1:
    else if (j < ncolb && a0 == 1)
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
