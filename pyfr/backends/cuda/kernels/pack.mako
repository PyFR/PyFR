# -*- coding: utf-8 -*-

__global__ void
pack_view(int nrow, int ncol, ${dtype}** vptr, int* vstri, ${dtype}* pmat,
          int ldp, int lds, int ldm)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nrow && j < ncol)
    {
        ${dtype}* ptr = vptr[i*ldp + j];
        int stride = vstri[i*lds + j];

    % for k in range(vlen):
        pmat[i*ldm + ${k}*ncol + j] = ptr[${k}*stride];
    % endfor
    }
}

__global__ void
unpack_view(int nrow, int ncol, ${dtype}** vptr, int* vstri,
            const ${dtype}* upmat, int ldp, int lds, int ldm)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nrow && j < ncol)
    {
        ${dtype}* ptr = vptr[i*ldp + j];
        int stride = vstri[i*lds + j];

    % for k in range(vlen):
        ptr[${k}*stride] = upmat[i*ldm + ${k}*ncol + j];
    % endfor
    }
}
