# -*- coding: utf-8 -*-

__global__ void
axnpby(int n, ${dtype}* y, ${dtype} beta,
       ${', '.join('const {0}* x{1}, {0} a{1}'.format(dtype, i)
         for i in range(n))})
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        ${dtype} axn = ${' + '.join('a{0}*x{0}[i]'.format(i)\
                         for i in range(n))};

        if (beta == 0.0)
            y[i] = axn;
        else if (beta == 1.0)
            y[i] += axn;
        else
            y[i] = beta*y[i] + axn;
    }
}
