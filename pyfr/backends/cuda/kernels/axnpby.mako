<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
axnpby(ixdtype_t ncolb, ixdtype_t ldim,
       fpdtype_t* __restrict__ x0,
       ${', '.join(f'const fpdtype_t* __restrict__ x{i}'
                   for i in range(1, nv))},
       ${', '.join(f'fpdtype_t a{i}' for i in range(nv))})
{
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    ixdtype_t j = ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x;
    ixdtype_t idx;

    if (j < ncolb && a0 == 0.0)
    {
    % for k in subdims:
        idx = i*ldim + SOA_IX(j, ${k}, ${ncola});
        x0[idx] = ${pyfr.dot('a{l}', 'x{l}[idx]', l=(1, nv))};
    % endfor
    }
    else if (j < ncolb && a0 == 1.0)
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
