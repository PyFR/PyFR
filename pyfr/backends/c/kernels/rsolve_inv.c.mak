# -*- coding: utf-8 -*-

<%include file='common.h.mak' />
<%include file='rsolve_inv_impl.h.mak' />
<%include file='views.h.mak' />

void
rsolve_inv_int(size_t ninters,
               ${dtype} **restrict ul_v,
               const int *restrict ul_vstri,
               ${dtype} **restrict ur_v,
               const int *restrict ur_vstri,
               const ${dtype} *restrict magpnorml,
               const ${dtype} *restrict magpnormr,
               const ${dtype} *restrict normpnorml)
{
    #pragma omp parallel for
    for (size_t iidx = 0; iidx < ninters; iidx++)
    {
        ${dtype} ul[${nvars}], ur[${nvars}];

        // Dereference the views into memory
        READ_VIEW(ul, ul_v, ul_vstri, iidx, ${nvars});
        READ_VIEW(ur, ur_v, ur_vstri, iidx, ${nvars});

        // Load the left normalized physical normal
        ${dtype} ptemp[${ndims}];
        for (int i = 0; i < ${ndims}; i++)
            ptemp[i] = normpnorml[ninters*i + iidx];

        // Perform the Riemann solve
        ${dtype} fn[${nvars}];
        rsolve_inv_impl(ul, ur, ptemp, fn);

        // Write out the fluxes into ul and ur
        for (int i = 0; i < ${nvars}; i++)
        {
            ul[i] =  magpnorml[iidx]*fn[i];
            ur[i] = -magpnormr[iidx]*fn[i];
        }

        // Copy back into the views
        WRITE_VIEW(ul_v, ul_vstri, ul, iidx, ${nvars});
        WRITE_VIEW(ur_v, ur_vstri, ur, iidx, ${nvars});
    }
}

void
rsolve_inv_mpi(size_t ninters,
               ${dtype} **restrict ul_v,
               const int *restrict ul_vstri,
               const ${dtype} *restrict ur_m,
               const ${dtype} *restrict magpnorml,
               const ${dtype} *restrict normpnorml)
{
    #pragma omp parallel for
    for (size_t iidx = 0; iidx < ninters; iidx++)
    {
        ${dtype} ptemp[${ndims}], ul[${nvars}], ur[${nvars}];

        // Load the left hand side (local) solution
        READ_VIEW(ul, ul_v, ul_vstri, iidx, ${nvars});

        // Load the left normalized physical normal
        for (int i = 0; i < ${ndims}; ++i)
            ptemp[i] = normpnorml[ninters*i + iidx];

        // Load the right hand (MPI) side solution matrix
        READ_MPIM(ur, ur_m, iidx, ninters, ${nvars});

        // Perform the Riemann solve
        ${dtype} fn[${nvars}];
        rsolve_inv_impl(ul, ur, ptemp, fn);

        // Write out the fluxes into ul
        for (int i = 0; i < ${nvars}; ++i)
            ul[i] = magpnorml[iidx]*fn[i];

        // Copy these back into the view
        WRITE_VIEW(ul_v, ul_vstri, ul, iidx, ${nvars});
    }
}

% if bctype:
<%include file='bc_impl.cu.mak' />

void
rsolve_inv_bc(size_t ninters,
              ${dtype} **restrict ul_v,
              const int *restrict ul_vstri,
              const ${dtype} *restrict magpnorml,
              const ${dtype} *restrict normpnorml)
{
    #pragma omp parallel for
    for (size_t iidx = 0; iidx < ninters; iidx++)
    {
        ${dtype} ul[${nvars}], ur[${nvars}];

        // Dereference the view into memory
        READ_VIEW(ul, ul_v, ul_vstri, iidx, ${nvars});

        // Compute the RHS (boundary) solution from this
        bc_u_impl(ul, ur);

        // Load the left normalized physical normal
        ${dtype} ptemp[${ndims}];
        for (int i = 0; i < ${ndims}; ++i)
            ptemp[i] = normpnorml[ninters*i + iidx];

        // Perform the Riemann solve
        ${dtype} fn[${nvars}];
        rsolve_inv_impl(ul, ur, ptemp, fn);

        // Write out the fluxes into ul
        for (int i = 0; i < ${nvars}; ++i)
            ul[i] =  magpnorml[iidx]*fn[i];

        // Copy back into the view
        WRITE_VIEW(ul_v, ul_vstri, ul, iidx, ${nvars});
    }
}
% endif
