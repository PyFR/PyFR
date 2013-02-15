# -*- coding: utf-8 -*-

<%namespace name='util' file='util.mak' />

<%include file='views.cu.mak' />
<%include file='flux_inv.cu.mak' />

/**
 * Rusanov Riemann solver from Z. J. Wang et al.
 */
inline __device__ void
rsolve_inv_rus(const ${dtype} ul[${nvars}],
               const ${dtype} ur[${nvars}],
               const ${dtype} pnorm[${ndims}],
               ${dtype} fcomm[${nvars}])
{
    // Compute the left and right fluxes + velocities and pressures
    ${dtype} fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    ${dtype} vl[${ndims}], vr[${ndims}];
    ${dtype} pl, pr;

    disf_inv(ul, fl, &pl, vl);
    disf_inv(ur, fr, &pr, vr);

    // Compute the speed/2
    ${dtype} a = sqrt(${0.25*gamma}*(pl + pr)/(ul[0] + ur[0]))
               + 0.25*fabs(${util.dot('pnorm[{0}]', 'vl[{0}] + vr[{0}]')});


    // Output
    for (int i = 0; i < ${nvars}; ++i)
        fcomm[i] = 0.5*${util.dot('pnorm[{0}]', 'fl[{0}][i] + fr[{0}][i]')}
                 + a*(ul[i] - ur[i]);

}

/**
 * HLL Riemann solver from Toro.
 */
inline __device__ void
rsolve_inv_hll(const ${dtype} ul[${nvars}],
               const ${dtype} ur[${nvars}],
               const ${dtype} pnorm[${ndims}],
               ${dtype} fcomm[${nvars}])
{
    // Compute the left and right fluxes + velocities and pressures
    ${dtype} fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    ${dtype} vl[${ndims}], vr[${ndims}];
    ${dtype} pl, pr;

    disf_inv(ul, fl, &pl, vl);
    disf_inv(ur, fr, &pr, vr);

    // Get the normal left and right velocities
    ${dtype} nvl = ${util.dot('pnorm[{0}]', 'vl[{0}]')};
    ${dtype} nvr = ${util.dot('pnorm[{0}]', 'vr[{0}]')};

    // Compute the enthalpies
    ${dtype} Hl = (ul[4] + pl)/ul[0];
    ${dtype} Hr = (ur[4] + pr)/ur[0];

    // Compute the Roe-averaged enthalpy
    ${dtype} H = (sqrt(ul[0])*Hl + sqrt(ur[0])*Hr)/(sqrt(ul[0]) + sqrt(ur[0]));

    // Compute the Roe-averaged velocity
    ${dtype} v = (sqrt(ul[0])*nvl + sqrt(ur[0])*nvr)/(sqrt(ul[0]) + sqrt(ur[0]));

    // Use these to compute the Roe-averaged sound speed
    ${dtype} a = sqrt(${gamma - 1.0}*(H - 0.5*v*v));

    // Compute Sl and Sr
    ${dtype} Sl = v - a;
    ${dtype} Sr = v + a;

    for (int i = 0; i < ${nvars}; ++i)
    {
        if (Sl > 0.0)
            fcomm[i] = ${util.dot('pnorm[{0}]', 'fl[{0}][i]')};
        else if (Sr < 0.0)
            fcomm[i] = ${util.dot('pnorm[{0}]', 'fr[{0}][i]')};
        else
            fcomm[i] = (${util.dot('pnorm[{0}]', 'Sr*fl[{0}][i] - Sl*fr[{0}][i]')}
                      + Sl*Sr*(ur[i] - ul[i]))
                     / (Sr - Sl);
    }
}

__global__ void
rsolve_inv_int(int ninters,
               ${dtype}** __restrict__ ul_v,
               const int* __restrict__ ul_vstri,
               ${dtype}** __restrict__ ur_v,
               const int* __restrict__ ur_vstri,
               const ${dtype}* __restrict__ magpnorml,
               const ${dtype}* __restrict__ magpnormr,
               const ${dtype}* __restrict__ normpnorml)
{
    int iidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (iidx < ninters)
    {
        ${dtype} ul[${nvars}], ur[${nvars}];

        // Dereference the views into memory
        READ_VIEW(ul, ul_v, ul_vstri, iidx, ${nvars});
        READ_VIEW(ur, ur_v, ur_vstri, iidx, ${nvars});

        // Load the left normalized physical normal
        ${dtype} ptemp[${ndims}];
        for (int i = 0; i < ${ndims}; ++i)
            ptemp[i] = normpnorml[ninters*i + iidx];

        // Perform the Riemann solve
        ${dtype} fn[${nvars}];
        rsolve_inv_rus(ul, ur, ptemp, fn);

        // Write out the fluxes into ul and ur
        for (int i = 0; i < ${nvars}; ++i)
        {
            ul[i] =  magpnorml[iidx]*fn[i];
            ur[i] = -magpnormr[iidx]*fn[i];
        }

        // Copy back into the views
        WRITE_VIEW(ul_v, ul_vstri, ul, iidx, ${nvars});
        WRITE_VIEW(ur_v, ur_vstri, ur, iidx, ${nvars});
    }
}

__global__ void
rsolve_inv_mpi(int ninters,
               ${dtype}** __restrict__ ul_v,
               const int* __restrict__ ul_vstri,
               const ${dtype}* __restrict__ ur_m,
               const ${dtype}* __restrict__ magpnorml,
               const ${dtype}* __restrict__ normpnorml)
{
    int iidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (iidx < ninters)
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
        ${rsinv}(ul, ur, ptemp, fn);

        // Write out the fluxes into ul
        for (int i = 0; i < ${nvars}; ++i)
            ul[i] = magpnorml[iidx]*fn[i];

        // Copy these back into the view
        WRITE_VIEW(ul_v, ul_vstri, ul, iidx, ${nvars});
    }
}
