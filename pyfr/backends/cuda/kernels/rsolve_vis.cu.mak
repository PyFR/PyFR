# -*- coding: utf-8 -*-

<%include file='idx_of.cu.mak' />
<%include file='views.cu.mak' />
<%include file='flux_vis.cu.mak' />

<%
# Special-case beta
if beta == -0.5:
    need_fl, need_fr = False, True
    nfexpr = ' + '.join('pnorml[{0}]*fr[{0}][i]'.format(k)
                        for k in range(ndims))
elif beta == 0.5:
    need_fl, need_fr = True, False
    nfexpr = ' + '.join('pnorml[{0}]*fl[{0}][i]'.format(k)
                        for k in range(ndims))
else:
    need_fl, need_fr = True, True
    nfexpr = ' + '.join('pnorml[{0}]*(fl[{0}][i]*(0.5 + {1})'
                                  ' + fr[{0}][i]*(0.5 - {1}))'.format(k, beta)
                        for k in range(ndims))

# Special-case tau
if tau == 0.0:
    need_ul, need_ur = need_fl, need_fr
else:
    need_ul, need_ur = True, True
    nfexpr += ' + {0}*(ul[i] - ur[i])'.format(tau)
%>

__global__ void
rsolve_ldg_vis_int(int ninters,
                   ${dtype}** __restrict__ ul_v,
                   const int* __restrict__ ul_vstri,
                   ${dtype}** __restrict__ gul_v,
                   const int* __restrict__ gul_vstri,
                   ${dtype}** __restrict__ ur_v,
                   const int* __restrict__ ur_vstri,
                   ${dtype}** __restrict__ gur_v,
                   const int* __restrict__ gur_vstri,
                   const ${dtype}* __restrict__ magpnorml,
                   const ${dtype}* __restrict__ magpnormr,
                   const ${dtype}* __restrict__ normpnorml)
{
    int iidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (iidx < ninters)
    {
    % if need_ul:
        // Load in the LHS soln
        ${dtype} ul[${nvars}];
        READ_VIEW(ul, ul_v, ul_vstri, iidx, ${nvars});
    % endif

    % if need_ur:
        // Load in the RHS soln
        ${dtype} ur[${nvars}];
        READ_VIEW(ur, ur_v, ur_vstri, iidx, ${nvars});
    % endif

    % if need_fl:
        // Compute the LHS flux
        ${dtype} gul[${ndims}][${nvars}], fl[${ndims}][${nvars}];
        READ_VIEW_V(gul, gul_v, gul_vstri, iidx, ninters, ${ndims}, ${nvars});
        disf_vis(ul, gul, fl, ${gamma}, ${mu}, ${pr});
    % endif

    % if need_fr:
        // Compute the RHS flux
        ${dtype} gur[${ndims}][${nvars}], fr[${ndims}][${nvars}];
        READ_VIEW_V(gur, gur_v, gur_vstri, iidx, ninters, ${ndims}, ${nvars});
        disf_vis(ur, gur, fr, ${gamma}, ${mu}, ${pr});
    % endif

        // Load the left normalized physical normal
        ${dtype} pnorml[${ndims}];
        for (int i = 0; i < ${ndims}; ++i)
            pnorml[i] = normpnorml[ninters*i + iidx];

        // Determine the common normal flux
        for (int i = 0; i < ${nvars}; ++i)
        {
            ${dtype} fcomm = ${nfexpr};

            ul[i] =  magpnorml[iidx]*fcomm;
            ur[i] = -magpnormr[iidx]*fcomm;
        }

        // Copy back into the views
        WRITE_VIEW(ul_v, ul_vstri, ul, iidx, ${nvars});
        WRITE_VIEW(ur_v, ur_vstri, ur, iidx, ${nvars});
    }
}

__global__ void
rsolve_ldg_vis_mpi(int ninters,
                   ${dtype}** __restrict__ ul_v,
                   const int* __restrict__ ul_vstri,
                   ${dtype}** __restrict__ gul_v,
                   const int* __restrict__ gul_vstri,
                   const ${dtype}* __restrict__ ur_m,
                   const ${dtype}* __restrict__ gur_m,
                   const ${dtype}* __restrict__ magpnorml,
                   const ${dtype}* __restrict__ normpnorml)
{
    int iidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (iidx < ninters)
    {
    % if need_ul:
        // Load in the LHS soln
        ${dtype} ul[${nvars}];
        READ_VIEW(ul, ul_v, ul_vstri, iidx, ${nvars});
    % endif

    % if need_ur:
        // Load in the RHS soln
        ${dtype} ur[${nvars}];
        for (int i = 0; i < ${nvars}; ++i)
            ur[i] = ur_m[ninters*i + iidx];
    % endif

    % if need_fl:
        // Compute the LHS flux
        ${dtype} gul[${ndims}][${nvars}], fl[${ndims}][${nvars}];
        READ_VIEW_V(gul, gul_v, gul_vstri, iidx, ninters, ${ndims}, ${nvars});
        disf_vis(ul, gul, fl, ${gamma}, ${mu}, ${pr});
    % endif

    % if need_fr:
        // Compute the RHS flux
        ${dtype} gur[${ndims}][${nvars}], fr[${ndims}][${nvars}];
        READ_MPIM_V(gur, gur_m, iidx, ninters, ${ndims}, ${nvars});
        disf_vis(ur, gur, fr, ${gamma}, ${mu}, ${pr});
    % endif

        // Load the left normalized physical normal
        ${dtype} pnorml[${ndims}];
        for (int i = 0; i < ${ndims}; ++i)
            pnorml[i] = normpnorml[ninters*i + iidx];

        // Determine the common normal flux
        for (int i = 0; i < ${nvars}; ++i)
            ul[i] = magpnorml[iidx]*(${nfexpr});

        // Copy back into the views
        WRITE_VIEW(ul_v, ul_vstri, ul, iidx, ${nvars});
    }
}
