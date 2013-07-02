# -*- coding: utf-8 -*-

<%namespace name='util' module='pyfr.backends.openmp.makoutil' />
<%include file='flux_inv_impl.h.mak' />

% if rsinv == 'rusanov':
/**
 * Rusanov Riemann solver from Z. J. Wang et al.
 */
static inline void
rsolve_inv_impl(const ${dtype} ul[${nvars}],
                const ${dtype} ur[${nvars}],
                const ${dtype} pnorm[${ndims}],
                ${dtype} fcomm[${nvars}])
{
    // Compute the left and right fluxes + velocities and pressures
    ${dtype} fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    ${dtype} vl[${ndims}], vr[${ndims}];
    ${dtype} pl, pr;

    disf_inv_impl(ul, fl, &pl, vl);
    disf_inv_impl(ur, fr, &pr, vr);

    // Compute the speed/2
    ${dtype} a = sqrt(${0.25*c['gamma']|f}*(pl + pr)/(ul[0] + ur[0]))
               + ${0.25|f}*fabs(${util.dot('pnorm[{0}]', 'vl[{0}] + vr[{0}]')});


    // Output
    for (int i = 0; i < ${nvars}; ++i)
        fcomm[i] = ${0.5|f}*${util.dot('pnorm[{0}]', 'fl[{0}][i] + fr[{0}][i]')}
                 + a*(ul[i] - ur[i]);

}
% elif rsinv == 'hll':
/**
 * HLL Riemann solver from Toro.
 */
static inline void
rsolve_inv_impl(const ${dtype} ul[${nvars}],
                const ${dtype} ur[${nvars}],
                const ${dtype} pnorm[${ndims}],
                ${dtype} fcomm[${nvars}])
{
    // Compute the left and right fluxes + velocities and pressures
    ${dtype} fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    ${dtype} vl[${ndims}], vr[${ndims}];
    ${dtype} pl, pr;

    disf_inv_impl(ul, fl, &pl, vl);
    disf_inv_impl(ur, fr, &pr, vr);

    // Get the normal left and right velocities
    ${dtype} nvl = ${util.dot('pnorm[{0}]', 'vl[{0}]')};
    ${dtype} nvr = ${util.dot('pnorm[{0}]', 'vr[{0}]')};

    // Compute the enthalpies
    ${dtype} Hl = (ul[${nvars - 1}] + pl)/ul[0];
    ${dtype} Hr = (ur[${nvars - 1}] + pr)/ur[0];

    // Compute the Roe-averaged enthalpy
    ${dtype} H = (sqrt(ul[0])*Hl + sqrt(ur[0])*Hr)/(sqrt(ul[0]) + sqrt(ur[0]));

    // Compute the Roe-averaged velocity
    ${dtype} v = (sqrt(ul[0])*nvl + sqrt(ur[0])*nvr)/(sqrt(ul[0]) + sqrt(ur[0]));

    // Use these to compute the Roe-averaged sound speed
    ${dtype} a = sqrt(${c['gamma'] - 1.0|f}*(H - ${0.5|f}*v*v));

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
% endif
