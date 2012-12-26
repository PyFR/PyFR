# -*- coding: utf-8 -*-

__global__ void
conu_int(int ninters,
         ${dtype}** __restrict__ ul_vin,
         const int* __restrict__ ul_vstri,
         ${dtype}** __restrict__ ur_vin,
         const int* __restrict__ ur_vstri,
         ${dtype}** __restrict__ ul_vout,
         ${dtype}** __restrict__ ur_vout,
         ${dtype} beta)
{
    int iidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (iidx < ninters)
    {
        int lstri = ul_vstri[iidx];
        int rstri = ur_vstri[iidx];

        for (int i = 0; i < ${nvars}; ++i)
        {
            ${dtype} l = ul_vin[iidx][lstri*i];
            ${dtype} r = ur_vin[iidx][rstri*i];

            ${dtype} con = r*(beta + 0.5) + l*(0.5 - beta);

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
         const ${dtype}* __restrict__ ur_m,
         ${dtype} beta)
{
    int iidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (iidx < ninters)
    {
        int lstri = ul_vstri[iidx];

        for (int i = 0; i < ${nvars}; ++i)
        {
            ${dtype} l = ul_vin[iidx][lstri*i];
            ${dtype} r = ur_m[ninters*i + iidx];

            ul_vout[iidx][lstri*i] = r*(beta + 0.5) + l*(0.5 - beta);
        }
    }
}
