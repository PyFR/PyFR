<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

kernel void
axnpby(constant ixdtype_t& nrow, constant ixdtype_t& ncolb,
       constant ixdtype_t& ldim, device fpdtype_t* x0,
       ${', '.join(f'device const fpdtype_t* x{i}' for i in range(1, nv))},
       ${', '.join(f'constant fpdtype_t& a{i}' for i in range(nv))},
       uint2 ji [[thread_position_in_grid]])
{
    ixdtype_t j = ji.x, i = ji.y;
    ixdtype_t idx;

    if (j < ncolb && a0 == 0)
    {
    % for k in subdims:
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] = ${pyfr.dot('a{l}', 'x{l}[idx]', l=(1, nv))};
    % endfor
    }
    else if (j < ncolb && a0 == 1)
    {
    % for k in subdims:
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] += ${pyfr.dot('a{l}', 'x{l}[idx]', l=(1, nv))};
    % endfor
    }
    else if (j < ncolb)
    {
    % for k in subdims:
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] = ${pyfr.dot('a{l}', 'x{l}[idx]', l=nv)};
    % endfor
    }
}
