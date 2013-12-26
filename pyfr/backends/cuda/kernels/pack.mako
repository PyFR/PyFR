# -*- coding: utf-8 -*-

__global__ void
pack_view(int nrow,
          int ncol,
          const ${dtype}* __restrict__ v,
          const int* __restrict__ vix,
          const int* __restrict__ vstri,
          ${dtype}* __restrict__  pmat)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i < nrow && j < ncol)
    {
        const ${dtype}* ptr = v + vix[i*ncol + j];
        int stride = vstri[i*ncol + j];

    % for k in range(vlen):
        pmat[i*ncol*${vlen} + ${k}*ncol + j] = ptr[${k}*stride];
    % endfor
    }
}
