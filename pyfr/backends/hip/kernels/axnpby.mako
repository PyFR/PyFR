# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
axnpby(int nrow, int ncolb, int ldim, fpdtype_t* __restrict__ x0,
       ${', '.join(f'const fpdtype_t* __restrict__ x{i}'
                   for i in range(1, nv))},
       ${', '.join(f'fpdtype_t a{i}' for i in range(nv))})
{
    int i = hipBlockIdx_y*hipBlockDim_y + hipThreadIdx_y;
    int j = hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x;
    int idx;

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
