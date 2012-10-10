# -*- coding: utf-8 -*-

% if mat_order == 'F':
#define IDX_OFF(i, j, ldim) ((i) + (j)*(ldim))
% else:
#define IDX_OFF(i, j, ldim) ((i)*(ldim) + (j))
% endif

__global__ void square(${mat_ctype} *mat, int nrow, int ncol, int ldim)
{
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i < nrow && j < ncol)
    {
        mat[IDX_OFF(i, j, ldim)] *= mat[IDX_OFF(i, j, ldim)];
    }
}
