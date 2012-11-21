# -*- coding: utf-8 -*-

#define IDX_OF(i, j, ldim) ((i)*(ldim) + (j))

__global__ void
pack_view(int nrow, int ncol, ${dtype} **vptr, int *vstri, ${dtype} *pmat,
          int ldp, int lds, int ldm)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nrow && j < ncol)
    {
        ${dtype} *ptr = vptr[IDX_OF(i, j, ldp)];
        uint stride = vstri[IDX_OF(i, j, lds)];

    % for k in xrange(vlen):
        pmat[IDX_OF(i, ${k}*ncol + j, ldm)] = ptr[${k}*stride];
    % endfor
    }
}

__global__ void
unpack_view(int nrow, int ncol, ${dtype} **vptr, int *vstri,
            const ${dtype} *upmat, int ldp, int lds, int ldm)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nrow && j < ncol)
    {
        ${dtype} *ptr = vptr[IDX_OF(i, j, ldp)];
        uint stride = vstri[IDX_OF(i, j, lds)];

    % for k in xrange(vlen):
        ptr[${k}*stride] = upmat[IDX_OF(i, ${k}*ncol + j, ldm)];
    % endfor
    }
}

#undef IDX_OF
