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
        % if c['ldg-beta'] == -0.5:
            ${dtype} con = ul_vin[iidx][lstri*i];
        % elif c['ldg-beta'] == 0.5:
            ${dtype} con = ur_vin[iidx][rstri*i];
        % else:
            ${dtype} con = ur_vin[iidx][rstri*i]*${0.5 + c['ldg-beta']|f}
                         + ul_vin[iidx][lstri*i]*${0.5 - c['ldg-beta']|f};
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
        % if c['ldg-beta'] == -0.5:
            ${dtype} con = ul_vin[iidx][lstri*i];
        % elif c['ldg-beta'] == 0.5:
            ${dtype} con = ur_m[ninters*i + iidx];
        % else:
            ${dtype} con = ur_m[ninters*i + iidx]*${0.5 + c['ldg-beta']|f}
                         + ul_vin[iidx][lstri*i]*${0.5 - c['ldg-beta']|f};
        % endif

            ul_vout[iidx][lstri*i] = con;
        }
    }
}

% if bctype:
<%include file='views' />
<%include file='bc_impl' />

__global__ void
conu_bc(int ninters,
        ${dtype}** __restrict__ ul_vin,
        const int* __restrict__ ul_vstri,
        ${dtype}** __restrict__ ul_vout)
{
    int iidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (iidx < ninters)
    {
        int lstri = ul_vstri[iidx];
        ${dtype} ul[${nvars}], ur[${nvars}];

        // Load in the LHS soln from the view
        READ_VIEW(ul, ul_vin, ul_vstri, iidx, ${nvars});

        // Compute the RHS (boundary) soln from the LHS
        bc_u_impl(ul, ur);

        for (int i = 0; i < ${nvars}; ++i)
            ul_vout[iidx][lstri*i] = ur[i]*(${0.5 + c['ldg-beta']|f})
                                   + ul[i]*(${0.5 - c['ldg-beta']|f});
    }
}
% endif
