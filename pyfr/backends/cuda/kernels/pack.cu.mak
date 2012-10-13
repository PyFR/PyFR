# -*- coding: utf-8 -*-

<% minor_dim = 'nrow' if view_order == 'F' else 'ncol' %>

% if view_order == 'F':
#define IDX_OFF(i, j, ldim) ((i) + (j)*(ldim))
% else:
#define IDX_OFF(i, j, ldim) ((i)*(ldim) + (j))
% endif

__global__ void pack(${mat_ctype} **view, int nrow, int ncol, int ldim,
                     ${mat_ctype} *packbuf)
{
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i < nrow && j < ncol)
    {
        packbuf[IDX_OFF(i, j, ${minor_dim})] = *view[IDX_OFF(i, j, ldim)];
    }
}

__global__ void unpack(${mat_ctype} **view, int nrow, int ncol, int ldim,
                       const ${mat_ctype} *unpackbuf)
{
    const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i < nrow && j < ncol)
    {
        *view[IDX_OFF(i, j, ldim)] = unpackbuf[IDX_OFF(i, j, ${minor_dim})];
    }
}
