<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

kernel void
axnpby(constant ixdtype_t& ncolb, constant ixdtype_t& ldim,
       device fpdtype_t* x0,
       ${', '.join(f'device const fpdtype_t* x{i}' for i in range(1, nv)) + ',' if nv > 1 else ''}
       ${', '.join(f'constant fpdtype_t& a{i}' for i in range(nv))},
       uint2 ji [[thread_position_in_grid]])
{
% if in_scale:
    const fpdtype_t _in[] = ${pyfr.carray(in_scale)};
% endif
% if out_scale:
    const fpdtype_t _out[] = ${pyfr.carray(out_scale)};
% endif
    ixdtype_t j = ji.x, i = ji.y;
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
