# -*- coding: utf-8 -*-

<%include file='common' />

void
conu_int(int ninters,
         ${dtype} **restrict ul_vin,
         const int *restrict ul_vstri,
         ${dtype} **restrict ur_vin,
         const int *restrict ur_vstri,
         ${dtype} **restrict ul_vout,
         ${dtype} **restrict ur_vout)
{
    #pragma omp parallel for
    for (int iidx = 0; iidx < ninters; iidx++)
    {
        int lstri = ul_vstri[iidx];
        int rstri = ur_vstri[iidx];

        for (int i = 0; i < ${nvars}; i++)
        {
            ${dtype} con = ur_vin[iidx][rstri*i]*${0.5 + c['ldg-beta']|f}
                         + ul_vin[iidx][lstri*i]*${0.5 - c['ldg-beta']|f};

            ul_vout[iidx][lstri*i] = con;
            ur_vout[iidx][rstri*i] = con;
        }
    }
}

void
conu_mpi(int ninters,
         ${dtype} **restrict ul_vin,
         const int *restrict ul_vstri,
         ${dtype} **restrict ul_vout,
         const ${dtype} *restrict ur_m)
{
    #pragma omp parallel for
    for (int iidx = 0; iidx < ninters; iidx++)
    {
        int lstri = ul_vstri[iidx];

        for (int i = 0; i < ${nvars}; i++)
        {
            ${dtype} con = ur_m[ninters*i + iidx]*${0.5 + c['ldg-beta']|f}
                         + ul_vin[iidx][lstri*i]*${0.5 - c['ldg-beta']|f};

            ul_vout[iidx][lstri*i] = con;
        }
    }
}

% if bctype:
<%include file='views' />
<%include file='bc_impl' />

void
conu_bc(int ninters,
        ${dtype} **restrict ul_vin,
        const int *restrict ul_vstri,
        ${dtype} **restrict ul_vout)
{
    #pragma omp parallel for
    for (int iidx = 0; iidx < ninters; iidx++)
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
