<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void
axnpby(ixdtype_t ncolb, ixdtype_t ldim,
       __global fpdtype_t* restrict x0,
       ${', '.join(f'__global const fpdtype_t* restrict x{i}'
                   for i in range(1, nv))},
       ${', '.join(f'fpdtype_t a{i}' for i in range(nv))})
{
    ixdtype_t i = get_global_id(1), j = get_global_id(0);
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
