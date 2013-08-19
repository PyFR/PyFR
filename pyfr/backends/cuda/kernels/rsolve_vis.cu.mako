# -*- coding: utf-8 -*-

<%include file='views.cu.mako' />
<%include file='rsolve_inv_impl.cu.mako' />
<%include file='flux_vis_impl.cu.mako' />

<%
beta, tau = c['ldg-beta'], c['ldg-tau']

# Special-case beta for the viscous flux expression
if beta == -0.5:
    need_fvl, need_fvr = False, True
    fvcomm = ' + '.join('pnorml[{0}]*fvr[{0}][i]'.format(k)
                        for k in range(ndims))
elif beta == 0.5:
    need_fvl, need_fvr = True, False
    fvcomm = ' + '.join('pnorml[{0}]*fvl[{0}][i]'.format(k)
                        for k in range(ndims))
else:
    need_fvl, need_fvr = True, True
    cvl, cvr = f(format(0.5 + beta)), f(format(0.5 - beta))
    fvcomm = ' + '.join('pnorml[{0}]*(fvl[{0}][i]*{1} + fvr[{0}][i]*{2})'
                        .format(k, cvl, cvr) for k in range(ndims))

# Special-case tau
if tau != 0.0:
    fvcomm += ' + ' + f(format(tau)) + '*(ul[i] - ur[i])'

# Encase
fvcomm = '(' + fvcomm + ')'
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
        ${dtype} ul[${nvars}], ur[${nvars}];

        // Load in the solutions
        READ_VIEW(ul, ul_v, ul_vstri, iidx, ${nvars});
        READ_VIEW(ur, ur_v, ur_vstri, iidx, ${nvars});

        // Load the left normalized physical normal
        ${dtype} pnorml[${ndims}];
        for (int i = 0; i < ${ndims}; ++i)
            pnorml[i] = normpnorml[ninters*i + iidx];

        // Perform a standard, inviscid Riemann solve
        ${dtype} ficomm[${nvars}];
        rsolve_inv_impl(ul, ur, pnorml, ficomm);

    % if need_fvl:
        ${dtype} gul[${ndims}][${nvars}];
        READ_VIEW_V(gul, gul_v, gul_vstri, iidx, ninters, ${ndims}, ${nvars});

        ${dtype} fvl[${ndims}][${nvars}] = {};
        disf_vis_impl_add(ul, gul, fvl);
    % endif

    % if need_fvr:
        ${dtype} gur[${ndims}][${nvars}];
        READ_VIEW_V(gur, gur_v, gur_vstri, iidx, ninters, ${ndims}, ${nvars});

        ${dtype} fvr[${ndims}][${nvars}] = {};
        disf_vis_impl_add(ur, gur, fvr);
    % endif

        for (int i = 0; i < ${nvars}; ++i)
        {
            // Evaluate the common viscous flux
            ${dtype} fvcommi = ${fvcomm};

            ul[i] =  magpnorml[iidx]*(ficomm[i] + fvcommi);
            ur[i] = -magpnormr[iidx]*(ficomm[i] + fvcommi);
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
        ${dtype} ul[${nvars}], ur[${nvars}];

        // Load in the solutions
        READ_VIEW(ul, ul_v, ul_vstri, iidx, ${nvars});
        READ_MPIM(ur, ur_m, iidx, ninters, ${nvars});

        // Load the left normalized physical normal
        ${dtype} pnorml[${ndims}];
        for (int i = 0; i < ${ndims}; ++i)
            pnorml[i] = normpnorml[ninters*i + iidx];

        // Perform a standard, inviscid Riemann solve
        ${dtype} ficomm[${nvars}];
        rsolve_inv_impl(ul, ur, pnorml, ficomm);

    % if need_fvl:
        ${dtype} gul[${ndims}][${nvars}];
        READ_VIEW_V(gul, gul_v, gul_vstri, iidx, ninters, ${ndims}, ${nvars});

        ${dtype} fvl[${ndims}][${nvars}] = {};
        disf_vis_impl_add(ul, gul, fvl);
    % endif

    % if need_fvr:
        ${dtype} gur[${ndims}][${nvars}];
        READ_MPIM_V(gur, gur_m, iidx, ninters, ${ndims}, ${nvars});

        ${dtype} fvr[${ndims}][${nvars}] = {};
        disf_vis_impl_add(ur, gur, fvr);
    % endif

        // Determine the common normal flux
        for (int i = 0; i < ${nvars}; ++i)
        {
            // Evaluate the common viscous flux
            ${dtype} fvcommi = ${fvcomm};

            ul[i] = magpnorml[iidx]*(ficomm[i] + fvcommi);
        }

        // Copy back into the views
        WRITE_VIEW(ul_v, ul_vstri, ul, iidx, ${nvars});
    }
}

% if bctype:
<%include file='bc_impl.cu.mak' />

__global__ void
rsolve_ldg_vis_bc(int ninters,
                  ${dtype}** __restrict__ ul_v,
                  const int* __restrict__ ul_vstri,
                  ${dtype}** __restrict__ gul_v,
                  const int* __restrict__ gul_vstri,
                  const ${dtype}* __restrict__ magpnorml,
                  const ${dtype}* __restrict__ normpnorml)
{
    int iidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (iidx < ninters)
    {
        ${dtype} ul[${nvars}], ur[${nvars}];

        // Load in the LHS solution
        READ_VIEW(ul, ul_v, ul_vstri, iidx, ${nvars});

        // Compute the RHS (boundary) solution from this
        bc_u_impl(ul, ur);

        // Load the left normalized physical normal
        ${dtype} pnorml[${ndims}];
        for (int i = 0; i < ${ndims}; ++i)
            pnorml[i] = normpnorml[ninters*i + iidx];

        // Perform a standard, inviscid Riemann solve
        ${dtype} ficomm[${nvars}];
        rsolve_inv_impl(ul, ur, pnorml, ficomm);

        // Load in the LHS gradient
        ${dtype} gul[${ndims}][${nvars}];
        READ_VIEW_V(gul, gul_v, gul_vstri, iidx, ninters, ${ndims}, ${nvars});

    % if need_fvl:
        ${dtype} fvl[${ndims}][${nvars}] = {};
        disf_vis_impl_add(ul, gul, fvl);
    % endif

    % if need_fvr:
        // Compute the RHS gradient from the LHS soln and gradient
        ${dtype} gur[${ndims}][${nvars}];
        bc_grad_u_impl(ul, gul, gur);

        ${dtype} fvr[${ndims}][${nvars}] = {};
        disf_vis_impl_add(ur, gur, fvr);
    % endif

        // Determine the common normal flux
        for (int i = 0; i < ${nvars}; ++i)
        {
            // Evaluate the common viscous flux
            ${dtype} fvcommi = ${fvcomm};

            ul[i] = magpnorml[iidx]*(ficomm[i] + fvcommi);
        }

        // Copy back into the view
        WRITE_VIEW(ul_v, ul_vstri, ul, iidx, ${nvars});
    }
}
% endif
