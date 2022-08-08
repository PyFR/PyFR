# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.entropy'/>

<% inf = 1e20 %>

<%pyfr:macro name='get_minima' params='u, dmin, pmin, emin'>
    fpdtype_t d, p, e;
    fpdtype_t ui[${nvars}];

    dmin = ${inf}; pmin = ${inf}; emin = ${inf};

    for (int i = 0; i < ${nupts}; i++)
    {
        % for j in range(nvars):
        ui[${j}] = u[i][${j}];
        % endfor

        ${pyfr.expand('compute_entropy', 'ui', 'd', 'p', 'e')};
        dmin = fmin(dmin, d); pmin = fmin(pmin, p); emin = fmin(emin, e);
    }

    // If enforcing constraints on fpts/qpts, compute minima on fpts/qpts
    % if con_fpts:
    for (int i = 0; i < ${nfpts}; i++)
    {
        for (int j = 0; j < ${nvars}; j++)
        {
            ui[j] = ${pyfr.dot('intfpts[i][{k}]*u[{k}][j]', k=nupts)};
        }

        ${pyfr.expand('compute_entropy', 'ui', 'd', 'p', 'e')};
        // Enforce only positivity constraints
        dmin = fmin(dmin, d); pmin = fmin(pmin, p);
    }
    % endif

    % if con_qpts:
    for (int i = 0; i < ${nqpts}; i++)
    {
        for (int j = 0; j < ${nvars}; j++)
        {
            ui[j] = ${pyfr.dot('intqpts[i][{k}]*u[{k}][j]', k=nupts)};
        }

        ${pyfr.expand('compute_entropy', 'ui', 'd', 'p', 'e')};
        // Enforce only positivity constraints
        dmin = fmin(dmin, d); pmin = fmin(pmin, p);
    }
    % endif
</%pyfr:macro>

<%pyfr:macro name='apply_filter' params='umodes, vdm, uf, f'>
    // Precompute filter factors per basis degree
    fpdtype_t ffac[${order + 1}];
    fpdtype_t v = ffac[0] = 1.0;

    // Utilize exp(-zeta*p**2) = pow(f, p**2)
    % for d in range(1, order + 1):
    v *= f;
    ffac[${d}] = v*v;
    % endfor

    // Compute filtered solution
    for (int uidx = 0; uidx < ${nupts}; uidx++)
    {
        for (int vidx = 0; vidx < ${nvars}; vidx++)
        {
            fpdtype_t tmp = 0.0;

            // Group terms by basis order
            % for d in range(order + 1):
            tmp += ffac[${d}]*(${' + '.join(f'vdm[uidx][{k}]*umodes[{k}][vidx]'
                                              for k, dd in enumerate(ubdegs) if dd == d)});
            % endfor

            uf[uidx][vidx] = tmp;
        }
    }
</%pyfr:macro>

<%pyfr:kernel name='entropyfilter' ndim='1'
              u='inout fpdtype_t[${str(nupts)}][${str(nvars)}]'
              entmin_int='inout fpdtype_t[${str(nfpts)}]'
              vdm='in broadcast fpdtype_t[${str(nupts)}][${str(nupts)}]'
              invvdm='in broadcast fpdtype_t[${str(nupts)}][${str(nupts)}]'
              intfpts='in broadcast fpdtype_t[${str(nfpts)}][${str(nupts)}]'
              intqpts='in broadcast fpdtype_t[${str(nqpts)}][${str(nupts)}]'>
    fpdtype_t dmin, pmin, emin;

    // Compute minimum entropy from current and adjacent elements
    fpdtype_t entmin = ${inf};
    for (int fidx = 0; fidx < ${nfpts}; fidx++)
    {
        entmin = fmin(entmin, entmin_int[fidx]);
    }

    // Check if solution is within bounds
    ${pyfr.expand('get_minima', 'u', 'dmin', 'pmin', 'emin')};

    // Filter if out of bounds
    if (dmin < ${d_min} || pmin < ${p_min} || emin < entmin - ${e_tol})
    {
        // Compute modal basis
        fpdtype_t umodes[${nupts}][${nvars}];

        for (int uidx = 0; uidx < ${nupts}; uidx++)
        {
            for (int vidx = 0; vidx < ${nvars}; vidx++)
            {
                umodes[uidx][vidx] = ${' + '.join(f'invvdm[uidx][{k}]*u[{k}][vidx]'
                                                   for k in range(nupts))};
            }
        }

        // Setup filter (solve for f = exp(-zeta))
        fpdtype_t f_low = 0.0;
        fpdtype_t f_high = 1.0;
        fpdtype_t f, f1, f2, f3;
        fpdtype_t dmin_low, pmin_low, emin_low;
        fpdtype_t dmin_high, pmin_high, emin_high;

        fpdtype_t uf[${nupts}][${nvars}] = {{0}};

        // Get bracketed guesses for regula falsi method;
        dmin_high = dmin; pmin_high = pmin; emin_high = emin; // Unfiltered minima were precomputed
        ${pyfr.expand('apply_filter', 'umodes', 'vdm', 'uf', 'f_low')};
        ${pyfr.expand('get_minima', 'uf', 'dmin_low', 'pmin_low', 'emin_low')};

        // Regularize constraints to be around zero
        dmin_low -= ${d_min}; dmin_high -= ${d_min};
        pmin_low -= ${p_min}; pmin_high -= ${p_min};
        emin_low -= entmin - ${e_tol}; emin_high -= entmin - ${e_tol};

        // Iterate filter strength with Illinois algorithm
        for (int iter = 0; iter < ${niters}; iter++)
        {
            // Compute new guess for each constraint (catch if root is not bracketed)
            f1 = (dmin_high > 0.0) ? f_high : (0.5*f_low*dmin_high - f_high*dmin_low)/(0.5*dmin_high - dmin_low + ${ill_tol});
            f2 = (pmin_high > 0.0) ? f_high : (0.5*f_low*pmin_high - f_high*pmin_low)/(0.5*pmin_high - pmin_low + ${ill_tol});
            f3 = (emin_high > 0.0) ? f_high : (0.5*f_low*emin_high - f_high*emin_low)/(0.5*emin_high - emin_low + ${ill_tol});

            // Compute guess as minima of individual constraints
            f = fmin(f1, fmin(f2, f3));

            // In case of bracketing failure (due to roundoff errors), revert to bisection
            f = ((f > f_high) || (f < f_low)) ? 0.5*(f_low + f_high) : f;

            ${pyfr.expand('apply_filter', 'umodes', 'vdm', 'uf', 'f')};
            ${pyfr.expand('get_minima', 'uf', 'dmin', 'pmin', 'emin')};

            // Compute new bracket and constraint values
            if (dmin < ${d_min} || pmin < ${p_min} || emin < entmin - ${e_tol})
            {
                f_high = f;
                dmin_high = dmin - ${d_min};
                pmin_high = pmin - ${p_min};
                emin_high = emin - (entmin - ${e_tol});
            }
            else
            {
                f_low = f;
                dmin_low = dmin - ${d_min};
                pmin_low = pmin - ${p_min};
                emin_low = emin - (entmin - ${e_tol});
            }

            // Stopping criteria
            if (f_high - f_low < ${f_tol})
            {
                break;
            }
        }

        // Apply filtered solution with bounds-preserving filter strength
        if (f == f_low)
        {
            // Bounds-preserving filtered solution computed in last iteration
            % for i,j in pyfr.ndrange(nupts, nvars):
            u[${i}][${j}] = uf[${i}][${j}];
            % endfor
        }
        else
        {
            ${pyfr.expand('apply_filter', 'umodes', 'vdm', 'u', 'f_low')};
            ${pyfr.expand('get_minima', 'u', 'pmin', 'pmin', 'emin')};
        }

        // Calculate minimum entropy from filtered solution
        emin = emin_low + entmin - ${e_tol};
    }

    // Set new minimum entropy within element for next stage
    for (int fidx = 0; fidx < ${nfpts}; fidx++)
    {
        entmin_int[fidx] = emin;
    }

</%pyfr:kernel>
