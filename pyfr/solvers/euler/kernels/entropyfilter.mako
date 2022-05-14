# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.entropy'/>

<% inf = 1e20%>
<% rf_tol = 1e-6%>

<%pyfr:macro name='get_minima' params='u, dmin, pmin, emin'>
    fpdtype_t d, p, e;
    fpdtype_t ui[${nvars}];

    dmin = ${inf}, pmin = ${inf}, emin = ${inf};

    for (int i = 0; i < ${nupts}; i++) {
        for (int j = 0; j < ${nvars}; j++) {
            ui[j] = u[i][j];
        }
        ${pyfr.expand('compute_entropy', 'ui', 'd', 'p', 'e')};
        dmin = fmin(dmin, d); pmin = fmin(pmin, p); emin = fmin(emin, e);
    }

    // If enforcing constraints on fpts/qpts, compute minima on fpts/qpts
    % if con_fpts:
    for (int i = 0; i < ${nfpts}; i++) {
        for (int j = 0; j < ${nvars}; j++) {
            // Interpolate to fpts
            ui[j] = 0.0;
            for (int k = 0; k < ${nupts}; k++) {
                ui[j] += intfpts[i][k]*u[k][j];
            }
        }

        ${pyfr.expand('compute_entropy', 'ui', 'd', 'p', 'e')};
        dmin = fmin(dmin, d); pmin = fmin(pmin, p); emin = fmin(emin, e);
    }
    % endif

    % if con_qpts:
    for (int i = 0; i < ${nqpts}; i++) {
        for (int j = 0; j < ${nvars}; j++) {
            // Interpolate to qpts
            ui[j] = 0.0;
            for (int k = 0; k < ${nupts}; k++) {
                ui[j] += intqpts[i][k]*u[k][j];
            }
        }

        ${pyfr.expand('compute_entropy', 'ui', 'd', 'p', 'e')};
        dmin = fmin(dmin, d); pmin = fmin(pmin, p); emin = fmin(emin, e);
    }
    % endif
</%pyfr:macro>

<%pyfr:macro name='apply_filter' params='umodes, ffac, uf, zeta'>
    // Precompute filter factors
    % for i in range(nupts):
    ffac[${i}] = exp(-zeta*${ubdegs2[i]});
    % endfor

    // Compute filtered solution
    for (int uidx = 0; uidx < ${nupts}; uidx++) {
        for (int vidx = 0; vidx < ${nvars}; vidx++) {
            uf[uidx][vidx] = 0.0;

            for (int midx = 0; midx < ${nupts}; midx++) {
                tmp = ffac[midx]*umodes[midx][vidx]; // Filtered mode
                uf[uidx][vidx] += vdm[uidx][midx]*tmp;
            }
        }
    }
</%pyfr:macro>

<%pyfr:kernel name='entropyfilter' ndim='1'
              u='inout fpdtype_t[${str(nupts)}][${str(nvars)}]'
              entmin='in fpdtype_t'
              vdm='in broadcast fpdtype_t[${str(nupts)}][${str(nupts)}]'
              invvdm='in broadcast fpdtype_t[${str(nupts)}][${str(nupts)}]'
              intfpts='in broadcast fpdtype_t[${str(nfpts)}][${str(nupts)}]'
              intqpts='in broadcast fpdtype_t[${str(nqpts)}][${str(nupts)}]'>
    fpdtype_t dmin, pmin, emin;

    // Check if solution is within bounds
    ${pyfr.expand('get_minima', 'u', 'dmin', 'pmin', 'emin')};

    // Filter if out of bounds
    if ((dmin < ${d_min}) || (pmin < ${p_min}) || (emin < entmin - ${e_atol})) {
        // Compute modal basis
        fpdtype_t umodes[${nupts}][${nvars}] = {{0}};

        for (int uidx = 0; uidx < ${nupts}; uidx++) {
            for (int vidx = 0; vidx < ${nvars}; vidx++) {
                for (int midx = 0; midx < ${nupts}; midx++) {
                    umodes[uidx][vidx] += invvdm[uidx][midx]*u[midx][vidx];
                }
            }
        }

        // Setup filter
        fpdtype_t zeta_low = 0.0;
        fpdtype_t zeta_high = ${zeta_max};
        fpdtype_t tmp, zeta, z1, z2, z3;
        fpdtype_t dmin_low, pmin_low, emin_low;
        fpdtype_t dmin_high, pmin_high, emin_high;

        fpdtype_t uf[${nupts}][${nvars}] = {{0}};
        fpdtype_t ffac[${nupts}] = {{0}};

        // Get bracketed guesses for regula falsi method;
        dmin_low = dmin; pmin_low = pmin; emin_low = emin; // Unfiltered minima were precomputed
        ${pyfr.expand('apply_filter', 'umodes', 'ffac', 'uf', 'zeta_high')};
        ${pyfr.expand('get_minima', 'uf', 'dmin_high', 'pmin_high', 'emin_high')};

        // Iterate filter strength
        for (int iter = 0; iter < ${niters}; iter++) {
            // Regularize constraints to be around zero
            dmin_low -= ${d_min}; dmin_high -= ${d_min};
            pmin_low -= ${p_min}; pmin_high -= ${p_min};
            emin_low -= entmin - ${e_atol}; emin_high -= entmin - ${e_atol};

            // Compute regular falsi for each constraint (catch if root is not bracketed)
            z1 = (dmin_low > 0.0) ? zeta_low : (zeta_low*dmin_high - zeta_high*dmin_low)/(dmin_high - dmin_low + ${rf_tol});
            z2 = (pmin_low > 0.0) ? zeta_low : (zeta_low*pmin_high - zeta_high*pmin_low)/(pmin_high - pmin_low + ${rf_tol});
            z3 = (emin_low > 0.0) ? zeta_low : (zeta_low*emin_high - zeta_high*emin_low)/(emin_high - emin_low + ${rf_tol});

            // Compute guess as maxima of individual constraints
            zeta = max(z1, max(z2, z3));

            // In case of bracketing failure (due to roundoff errors), revert to bisection
            zeta = ((zeta > zeta_high) || (zeta < zeta_low)) ? 0.5*(zeta_low + zeta_high): zeta;

            ${pyfr.expand('apply_filter', 'umodes', 'ffac', 'uf', 'zeta')};
            ${pyfr.expand('get_minima', 'uf', 'dmin', 'pmin', 'emin')};

            // Compute new bracket and constraint values
            if ((dmin < ${d_min}) || (pmin < ${p_min}) || (emin < entmin - ${e_atol})) {
                zeta_low = zeta;
                dmin_low = dmin;
                pmin_low = pmin;
                emin_low = emin;
            }
            else {
                zeta_high = zeta;
                dmin_high = dmin;
                pmin_high = pmin;
                emin_high = emin;
            }
        }

        // Apply filtered solution with bounds-preserving filter strength
        zeta = zeta_high;
        ${pyfr.expand('apply_filter', 'umodes', 'ffac', 'u', 'zeta')};
    }
    
</%pyfr:kernel>
