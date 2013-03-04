# -*- coding: utf-8 -*-

<%namespace name='util' module='pyfr.backends.cuda.makoutil' />

__global__ void
axnpby(int n, ${dtype}* y, ${dtype} beta,
       ${', '.join('const {0}* x{1}, {0} a{1}'.format(dtype, i)
         for i in range(n))})
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        ${dtype} axn = ${util.dot('a{0}', 'x{0}[i]', len='n')};

        if (beta == 0.0)
            y[i] = axn;
        else if (beta == 1.0)
            y[i] += axn;
        else
            y[i] = beta*y[i] + axn;
    }
}
