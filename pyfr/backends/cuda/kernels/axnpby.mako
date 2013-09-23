# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
axnpby(int n, ${dtype}* y, ${dtype} beta,
       ${', '.join('const {0}* x{1}, {0} a{1}'.format(dtype, i)
         for i in range(n))})
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < n)
    {
        ${dtype} axn = ${pyfr.dot('a{j}', 'x{j}[i]', j=n)};

        if (beta == 0.0)
            y[i] = axn;
        else if (beta == 1.0)
            y[i] += axn;
        else
            y[i] = beta*y[i] + axn;
    }
}
