# -*- coding: utf-8 -*-

__global__ void
conu_int(int ninters,
         ${dtype}** __restrict__ ul_vin,
         const int* __restrict__ ul_vstri,
         ${dtype}** __restrict__ ur_vin,
         const int* __restrict__ ur_vstri,
         ${dtype}** __restrict__ ul_vout,
         ${dtype}** __restrict__ ur_vout)
{
    int iidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (iidx < ninters)
    {
        int lstri = ul_vstri[iidx];
        int rstri = ur_vstri[iidx];

        for (int i = 0; i < ${nvars}; ++i)
        {
        % if beta == -0.5:
            ${dtype} con = ul_vin[iidx][lstri*i];
        % elif beta == 0.5:
            ${dtype} con = ur_vin[iidx][rstri*i];
        % else:
            ${dtype} con = ur_vin[iidx][rstri*i]*(${beta} + 0.5)
                         + ul_vin[iidx][lstri*i]*(0.5 - ${beta});
        % endif

            ul_vout[iidx][lstri*i] = con;
            ur_vout[iidx][rstri*i] = con;
        }
    }
}

__global__ void
conu_mpi(int ninters,
         ${dtype}** __restrict__ ul_vin,
         const int* __restrict__ ul_vstri,
         ${dtype}** __restrict__ ul_vout,
         const ${dtype}* __restrict__ ur_m)
{
    int iidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (iidx < ninters)
    {
        int lstri = ul_vstri[iidx];

        for (int i = 0; i < ${nvars}; ++i)
        {
        % if beta == -0.5:
            ${dtype} con = ul_vin[iidx][lstri*i];
        % elif beta == 0.5:
            ${dtype} con = ur_m[ninters*i + iidx];
        % else:
            ${dtype} con = ur_m[ninters*i + iidx]*(${beta} + 0.5)
                         + ul_vin[iidx][lstri*i]*(0.5 - ${beta});
        % endif

            ul_vout[iidx][lstri*i] = con;
        }
    }
}
