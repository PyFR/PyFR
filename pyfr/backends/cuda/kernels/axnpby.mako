# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
axnpby(int nrow, int ncolb, int ldim, int lsdim,
       ${', '.join('fpdtype_t* __restrict__ x' + str(i) for i in range(nv))},
       ${', '.join('fpdtype_t a' + str(i) for i in range(nv))})
{
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int idx;

    % for k in subdims:
    if (j < ncolb && a0 == 0.0)
        for (int i = 0; i < nrow; ++i)
        {
            idx = i*ldim + ${k}*lsdim + j;
            x0[idx] = ${pyfr.dot('a{l}', 'x{l}[idx]', l=(1, nv))};
        }
    else if (j < ncolb && a0 == 1.0)
        for (int i = 0; i < nrow; ++i)
        {
            idx = i*ldim + ${k}*lsdim + j;
            x0[idx] += ${pyfr.dot('a{l}', 'x{l}[idx]', l=(1, nv))};
        }
    else if (j < ncolb)
        for (int i = 0; i < nrow; ++i)
        {
            idx = i*ldim + ${k}*lsdim + j;
            x0[idx] = ${pyfr.dot('a{l}', 'x{l}[idx]', l=nv)};
        }
    % endfor
}
