# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
axnpby(int n, fpdtype_t* y, fpdtype_t beta,
       ${', '.join('const fpdtype_t* x{0}, fpdtype_t a{0}'.format(i)
                   for i in range(n))})
{
    int strt = blockIdx.x*blockDim.x + threadIdx.x;
    int incr = gridDim.x*blockDim.x;

    for (int i = strt; i < n; i += incr)
    {
        fpdtype_t axn = ${pyfr.dot('a{j}', 'x{j}[i]', j=n)};

        if (beta == 0.0)
            y[i] = axn;
        else if (beta == 1.0)
            y[i] += axn;
        else
            y[i] = beta*y[i] + axn;
    }
}
