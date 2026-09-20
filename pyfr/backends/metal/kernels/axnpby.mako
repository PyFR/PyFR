<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

struct _sargs_t
{
    ixdtype_t ncolb, ldim;
    fpdtype_t a[${nv}];
};

kernel void
axnpby(constant _sargs_t& _sa,
       device fpdtype_t* x0,
       ${', '.join(f'device const fpdtype_t* x{i}' for i in range(1, nv)) + ',' if nv > 1 else ''}
       uint2 ji [[thread_position_in_grid]])
{
    const ixdtype_t ncolb = _sa.ncolb, ldim = _sa.ldim;
% for i in range(nv):
    fpdtype_t a${i} = _sa.a[${i}];
% endfor
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
        idx = i*ldim + ${pyfr.soa_ix('j', k, ncola)};
        x0[idx] = ${pyfr.axnpby_expr(k, 'idx', 1, nv=nv, in_scale_idxs=in_scale_idxs, out_scale=out_scale)};
    % endfor
    }
% if nv > 1:
    else if (j < ncolb && a0 == 1)
    {
    % for k in range(ncola):
        idx = i*ldim + ${pyfr.soa_ix('j', k, ncola)};
        x0[idx] += ${pyfr.axnpby_expr(k, 'idx', 1, nv=nv, in_scale_idxs=in_scale_idxs, out_scale=out_scale)};
    % endfor
    }
% endif
    else if (j < ncolb)
    {
    % for k in range(ncola):
        idx = i*ldim + ${pyfr.soa_ix('j', k, ncola)};
        x0[idx] = ${pyfr.axnpby_expr(k, 'idx', 0, nv=nv, in_scale_idxs=in_scale_idxs, out_scale=out_scale)};
    % endfor
    }
}
