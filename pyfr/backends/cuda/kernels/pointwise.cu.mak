# -*- coding: utf-8 -*-

<%include file='idx_of.cu.mak' />

#define READ_VIEW(dst, src_v, src_vstri, vidx, vstriidx, nvec) \
    for (int _i = 0; _i < nvec; ++_i)                          \
        dst[_i] = src_v[vidx][src_vstri[vstriidx]*_i]

#define WRITE_VIEW(dst_v, dst_vstri, src, vidx, vstriidx, nvec) \
    for (int _i = 0; _i < nvec; ++_i)                           \
        dst_v[vidx][dst_vstri[vstriidx]*_i] = src[_i]

% if ndims == 3:
/**
 * Compute the inviscid flux.
 */
inline __device__ void
disf_inv(const ${dtype} s[5], ${dtype} f[5][3],
         ${dtype} gamma, ${dtype}* pout, ${dtype} vout[3])
{
    ${dtype} rho = s[0], rhou = s[1], rhov = s[2], rhow = s[3], E = s[4];

    ${dtype} invrho = 1.0/rho;
    ${dtype} u = invrho*rhou, v = invrho*rhov, w = invrho*rhow;

    // Compute the pressure
    ${dtype} p = (gamma - 1.0)*(E - 0.5*(rhou*u + rhov*v + rhow*w));

    f[0][0] = rhou;         f[0][1] = rhov;         f[0][2] = rhow;

    f[1][0] = rhou*u + p;   f[1][1] = rhov*u;       f[1][2] = rhow*u;
    f[2][0] = rhou*v;       f[2][1] = rhov*v + p;   f[2][2] = rhow*v;
    f[3][0] = rhou*w;       f[3][1] = rhov*w;       f[3][2] = rhow*w + p;

    f[4][0] = (E + p)*u;    f[4][1] = (E + p)*v;    f[4][2] = (E + p)*w;

    if (pout != NULL)
    {
        *pout = p;
    }

    if (vout != NULL)
    {
        vout[0] = u; vout[1] = v; vout[2] = w;
    }
}
% endif

/**
 * Computes the transformed inviscid flux.
 */
__global__ void
tdisf_inv(int nupts, int neles,
           const ${dtype}* __restrict__ u,
           const ${dtype}* __restrict__ smats,
           ${dtype}* __restrict__ f,
           ${dtype} gamma, int ldu, int lds, int ldf)
{
    int eidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (eidx < neles)
    {
        ${dtype} uin[${nvars}], ftmp[${nvars}][${ndims}];

        for (int uidx = 0; uidx < nupts; ++uidx)
        {
            // Load in the soln
            for (int i = 0; i < ${nvars}; ++i)
                uin[i] = u[U_IDX_OF(uidx, eidx, i, neles, ldu)];

            // Compute the flux
            disf_inv(uin, ftmp, gamma, NULL, NULL);

            // Transform and store
            for (int i = 0; i < ${ndims}; ++i)
            {
            % for k in range(ndims):
                ${dtype} s${k} = smats[SMAT_IDX_OF(uidx, eidx, i, ${k}, neles, lds)];
            % endfor

                for (int j = 0; j < ${nvars}; ++j)
                {
                    int fidx = F_IDX_OF(uidx, eidx, i, j, nupts, neles, ldf);
                    f[fidx] = ${' + '.join('s{0}*ftmp[j][{0}]'.format(k)\
                                for k in range(ndims))};
                }
            }
        }
    }
}

inline __device__ void
rsolve_rus_inv(const ${dtype} ul[${nvars}],
               const ${dtype} ur[${nvars}],
               const ${dtype} pnorm[${ndims}],
               ${dtype} fcomm[${nvars}],
               ${dtype} gamma)
{
    // Compute the left and right fluxes + velocities and pressures
    ${dtype} fl[${nvars}][${ndims}], fr[${nvars}][${ndims}];
    ${dtype} vl[${ndims}], vr[${ndims}];
    ${dtype} pl, pr;

    disf_inv(ul, fl, gamma, &pl, vl);
    disf_inv(ur, fr, gamma, &pr, vr);

    // Compute the speed
    ${dtype} a = sqrt(gamma*(pl + pr)/(ul[0] + ur[0]))
         + 0.5 * fabs(${' + '.join('pnorm[{0}]*(vl[{0}] + vr[{0}])'.format(k)\
                        for k in range(ndims))});

    // Output
    for (int i = 0; i < ${nvars}; ++i)
        fcomm[i] = 0.5 * ((${' + '.join('pnorm[{0}]*(fl[i][{0}] + fr[i][{0}])'.format(k)\
                             for k in range(ndims))})
                        - a*(ur[i] - ul[i]));

}


__global__ void
rsolve_rus_inv_int(int ninters,
                   ${dtype}** __restrict__ ul_v,
                   const int* __restrict__ ul_vstri,
                   ${dtype}** __restrict__ ur_v,
                   const int* __restrict__ ur_vstri,
                   const ${dtype}* __restrict__ magpnorml,
                   const ${dtype}* __restrict__ magpnormr,
                   const ${dtype}* __restrict__ normpnorml,
                   ${dtype} gamma)
{
    int iidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (iidx < ninters)
    {
        ${dtype} ul[${nvars}], ur[${nvars}];

        // Dereference the views into memory
        READ_VIEW(ul, ul_v, ul_vstri, iidx, iidx, ${nvars});
        READ_VIEW(ur, ur_v, ur_vstri, iidx, iidx, ${nvars});

        // Load the left normalized physical normal
        ${dtype} ptemp[${ndims}];
        for (int i = 0; i < ${ndims}; ++i)
            ptemp[i] = normpnorml[ninters*i + iidx];

        // Perform the Riemann solve
        ${dtype} fn[${nvars}];
        rsolve_rus_inv(ul, ur, ptemp, fn, gamma);

        // Write out the fluxes into ul and ur
        for (int i = 0; i < ${nvars}; ++i)
        {
            ul[i] =  magpnorml[iidx]*fn[i];
            ur[i] = -magpnormr[iidx]*fn[i];
        }

        // Copy back into the views
        WRITE_VIEW(ul_v, ul_vstri, ul, iidx, iidx, ${nvars});
        WRITE_VIEW(ur_v, ur_vstri, ur, iidx, iidx, ${nvars});
    }
}

__global__ void
rsolve_rus_inv_mpi(int ninters,
                   ${dtype}** __restrict__ ul_v,
                   const int* __restrict__ ul_vstri,
                   const ${dtype}* __restrict__ ur_m,
                   const ${dtype}* __restrict__ magpnorml,
                   const ${dtype}* __restrict__ normpnorml,
                   ${dtype} gamma)
{
    int iidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (iidx < ninters)
    {
        ${dtype} ptemp[${ndims}], ul[${nvars}], ur[${nvars}];

        // Load the left hand side (local) solution
        READ_VIEW(ul, ul_v, ul_vstri, iidx, iidx, ${nvars});

        // Load the left normalized physical normal
        for (int i = 0; i < ${ndims}; ++i)
            ptemp[i] = normpnorml[ninters*i + iidx];

        // Load the right hand (MPI) side solution matrix
        for (int i = 0; i < ${nvars}; ++i)
            ur[i] = ur_m[ninters*i + iidx];

        // Perform the Riemann solve
        ${dtype} fn[${nvars}];
        rsolve_rus_inv(ul, ur, ptemp, fn, gamma);

        // Write out the fluxes into ul
        for (int i = 0; i < ${nvars}; ++i)
            ul[i] = magpnorml[iidx]*fn[i];

        // Copy these back into the view
        WRITE_VIEW(ul_v, ul_vstri, ul, iidx, iidx, ${nvars});
    }
}

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
