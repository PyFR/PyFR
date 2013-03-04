# -*- coding: utf-8 -*-

<%include file='idx_of.cu.mak' />

__global__ void
negdivconf(int nupts, int neles,
           ${dtype}* __restrict__ tdivtconf,
           const ${dtype}* __restrict__ rcpdjac,
           int ldt, int ldr)
{
    int eidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (eidx < neles)
    {
        for (int uidx = 0; uidx < nupts; ++uidx)
        {
            ${dtype} s = -rcpdjac[IDX_OF(uidx, eidx, ldr)];

        % for i in range(nvars):
            tdivtconf[U_IDX_OF(uidx, eidx, ${i}, neles, ldt)] *= s;
        % endfor
        }
    }
}
