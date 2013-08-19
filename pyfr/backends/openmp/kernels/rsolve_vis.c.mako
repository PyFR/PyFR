# -*- coding: utf-8 -*-

<%include file='common.h.mako' />
<%include file='rsolve_inv_impl.h.mako' />
<%include file='flux_vis_impl.h.mako' />
<%include file='views.h.mako' />

<%
beta, tau = c['ldg-beta'], c['ldg-tau']

cvl, cvr = f(format(0.5 + beta)), f(format(0.5 - beta))
fvcomm = ' + '.join('pnorml[{0}]*(fvl[{0}][i]*{1} + fvr[{0}][i]*{2})'
                    .format(k, cvl, cvr) for k in range(ndims))


fvcomm += ' + ' + f(format(tau)) + '*(ul[i] - ur[i])'

# Encase
fvcomm = '(' + fvcomm + ')'
%>

void
rsolve_ldg_vis_int(size_t ninters,
                   ${dtype} **restrict ul_v,
                   const int *restrict ul_vstri,
                   ${dtype} **restrict gul_v,
                   const int *restrict gul_vstri,
                   ${dtype} **restrict ur_v,
                   const int *restrict ur_vstri,
                   ${dtype} **restrict gur_v,
                   const int *restrict gur_vstri,
                   const ${dtype} *restrict magpnorml,
                   const ${dtype} *restrict magpnormr,
                   const ${dtype} *restrict normpnorml)
{
    #pragma omp parallel for
    for (size_t iidx = 0; iidx < ninters; iidx++)
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

        ${dtype} gul[${ndims}][${nvars}];
        READ_VIEW_V(gul, gul_v, gul_vstri, iidx, ninters, ${ndims}, ${nvars});

        ${dtype} fvl[${ndims}][${nvars}] = {};
        disf_vis_impl_add(ul, gul, fvl);

        ${dtype} gur[${ndims}][${nvars}];
        READ_VIEW_V(gur, gur_v, gur_vstri, iidx, ninters, ${ndims}, ${nvars});

        ${dtype} fvr[${ndims}][${nvars}] = {};
        disf_vis_impl_add(ur, gur, fvr);

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

void
rsolve_ldg_vis_mpi(size_t ninters,
                   ${dtype} **restrict ul_v,
                   const int *restrict ul_vstri,
                   ${dtype} **restrict gul_v,
                   const int *restrict gul_vstri,
                   const ${dtype} *restrict ur_m,
                   const ${dtype} *restrict gur_m,
                   const ${dtype} *restrict magpnorml,
                   const ${dtype} *restrict normpnorml)
{
    #pragma omp parallel for
    for (size_t iidx = 0; iidx < ninters; iidx++)
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

        ${dtype} gul[${ndims}][${nvars}];
        READ_VIEW_V(gul, gul_v, gul_vstri, iidx, ninters, ${ndims}, ${nvars});

        ${dtype} fvl[${ndims}][${nvars}] = {};
        disf_vis_impl_add(ul, gul, fvl);

        ${dtype} gur[${ndims}][${nvars}];
        READ_MPIM_V(gur, gur_m, iidx, ninters, ${ndims}, ${nvars});

        ${dtype} fvr[${ndims}][${nvars}] = {};
        disf_vis_impl_add(ur, gur, fvr);

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
<%include file='bc_impl.h.mako' />

void
rsolve_ldg_vis_bc(size_t ninters,
                  ${dtype} **restrict ul_v,
                  const int *restrict ul_vstri,
                  ${dtype} **restrict gul_v,
                  const int *restrict gul_vstri,
                  const ${dtype} *restrict magpnorml,
                  const ${dtype} *restrict normpnorml)
{
    #pragma omp parallel for
    for (size_t iidx = 0; iidx < ninters; iidx++)
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

        ${dtype} fvl[${ndims}][${nvars}] = {};
        disf_vis_impl_add(ul, gul, fvl);

        // Compute the RHS gradient from the LHS soln and gradient
        ${dtype} gur[${ndims}][${nvars}];
        bc_grad_u_impl(ul, gul, gur);

        ${dtype} fvr[${ndims}][${nvars}] = {};
        disf_vis_impl_add(ur, gur, fvr);

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
