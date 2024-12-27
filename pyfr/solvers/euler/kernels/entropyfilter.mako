<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.entropy'/>

<%pyfr:macro name='get_minima' params='u, m0, dmin, pmin, emin'>
    fpdtype_t d, p, e;
    fpdtype_t ui[${nvars}];

    dmin = ${fpdtype_max}; pmin = ${fpdtype_max}; emin = ${fpdtype_max};

    for (int i = 0; i < ${nupts}; i++)
    {
    % for j in range(nvars):
        ui[${j}] = u[i][${j}];
    % endfor

        ${pyfr.expand('compute_entropy', 'ui', 'd', 'p', 'e')};
        dmin = fmin(dmin, d); pmin = fmin(pmin, p); emin = fmin(emin, e);
    }

    % if not fpts_in_upts:
    fpdtype_t uf[${nvars}];
    for (int fidx = 0; fidx < ${nfpts}; fidx++)
    {
        % for vidx in range(nvars):
        uf[${vidx}] = ${pyfr.dot('m0[fidx][{k}]', f'u[{{k}}][{vidx}]', k=nupts)};
        % endfor

        ${pyfr.expand('compute_entropy', 'uf', 'd', 'p', 'e')};
        dmin = fmin(dmin, d); pmin = fmin(pmin, p); emin = fmin(emin, e);
    }
    % endif
</%pyfr:macro>

<%pyfr:macro name='apply_filter_full' params='umodes, vdm, uf, f'>
    // Precompute filter factors per basis degree
    fpdtype_t ffac[${order + 1}];
    fpdtype_t v = ffac[0] = 1.0;

    // Utilize exp(-zeta*(p+1)**2) = exp(-zeta*p**2)*exp(-2*zeta*p)*exp(-zeta)
% for d in range(1, order + 1):
    ffac[${d}] = ffac[${d - 1}]*v*v*f;
    v *= f;
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

<%pyfr:macro name='apply_filter_single' params='up, f, d, p, e'>
    // Start accumulation
    fpdtype_t ui[${nvars}];
% for vidx in range(nvars):
    ui[${vidx}] = up[0][${vidx}];
% endfor

    // Apply filter to local value
    fpdtype_t v = 1.0, v2 = 1.0;
    for (int pidx = 1; pidx < ${order+1}; pidx++)
    {
        // Utilize exp(-zeta*(p+1)**2) = exp(-zeta*p**2)*exp(-2*zeta*p)*exp(-zeta)
        v2 *= v*v*f;
        v *= f;

        % for vidx in range(nvars):
        ui[${vidx}] += v2*up[pidx][${vidx}];
        % endfor
    }

    ${pyfr.expand('compute_entropy', 'ui', 'd', 'p', 'e' )};
</%pyfr:macro>

<%pyfr:kernel name='entropyfilter' ndim='1'
              u='inout fpdtype_t[${str(nupts)}][${str(nvars)}]'
              entmin_int='inout fpdtype_t[${str(nfaces)}]'
              vdm='in broadcast fpdtype_t[${str(nefpts)}][${str(nupts)}]'
              invvdm='in broadcast fpdtype_t[${str(nupts)}][${str(nupts)}]'
              m0='in broadcast fpdtype_t[${str(nfpts)}][${str(nupts)}]'>
    fpdtype_t dmin, pmin, emin;

    // Compute minimum entropy from current and adjacent elements
    fpdtype_t entmin = ${fpdtype_max};
    for (int fidx = 0; fidx < ${nfaces}; fidx++) entmin = fmin(entmin, entmin_int[fidx]);

    // Check if solution is within bounds
    ${pyfr.expand('get_minima', 'u', 'm0', 'dmin', 'pmin', 'emin')};

    // Filter if out of bounds
    if (dmin < ${d_min} || pmin < ${p_min} || emin < entmin - ${e_tol})
    {
        % if linearise:
        // Compute mean quantities
        fpdtype_t uavg[${nvars}], davg, pavg, eavg;
        % for vidx in range(nvars):
        uavg[${vidx}] = ${' + '.join(f'{jx}*u[{j}][{vidx}]'
                                     for j, jx in enumerate(meanwts) if jx != 0)};
        % endfor

        ${pyfr.expand('compute_entropy', 'uavg', 'davg', 'pavg', 'eavg')};

        // Apply density, pressure, and entropy limiting sequentially
        fpdtype_t alpha;
        % for (fvar, bound) in [('d', d_min), ('p', p_min), ('e', f'entmin - {e_tol}')]:
        if (${fvar}min < ${bound}) 
        {
            alpha = (${fvar}min - (${bound}))/(${fvar}min - ${fvar}avg);
            alpha = fmin(fmax(alpha, 0.0), 1.0);

            % for uidx, vidx in pyfr.ndrange(nupts, 1 if fvar == 'd' else nvars):
            u[${uidx}][${vidx}] += alpha*(uavg[${vidx}] - u[${uidx}][${vidx}]);
            % endfor

            ${pyfr.expand('get_minima', 'u', 'm0', 'dmin', 'pmin', 'emin')};
        }
        % endfor
        % else:
        // Compute modal basis
        fpdtype_t umodes[${nupts}][${nvars}];
        for (int uidx = 0; uidx < ${nupts}; uidx++)
        {
            for (int vidx = 0; vidx < ${nvars}; vidx++)
            {
                umodes[uidx][vidx] = ${pyfr.dot('invvdm[uidx][{k}]', 'u[{k}][vidx]', k=nupts)};
            }
        }

        // Setup filter (solve for f = exp(-zeta))
        fpdtype_t f = 1.0;
        fpdtype_t f_low, f_high, fnew;

        fpdtype_t d, p, e;

        // Compute f on a rolling basis per solution point
        fpdtype_t up[${order+1}][${nvars}];

        for (int uidx = 0; uidx < ${nefpts}; uidx++)
        {
            // Group nodal contributions by common filter factor
            % for pidx, vidx in pyfr.ndrange(order+1, nvars):
            up[${pidx}][${vidx}] = (${' + '.join(f'vdm[uidx][{k}]*umodes[{k}][{vidx}]'
                                                   for k, dd in enumerate(ubdegs) if dd == pidx)});
            % endfor

            // Compute constraints with current minimum f value
            ${pyfr.expand('apply_filter_single', 'up', 'f', 'd', 'p', 'e')};

            // Update f if constraints aren't satisfied
            if (d < ${d_min} || p < ${p_min} || e < entmin - ${e_tol})
            {
                // Set root-finding interval
                f_high = f;
                f_low = 0.0;

                // Iterate filter strength with bisection algorithm
                for (int iter = 0; iter < ${niters} && f_high - f_low > ${f_tol}; iter++)
                {
                    // Compute new guess using bisection
                    fnew = 0.5*(f_low + f_high);

                    // Compute filtered state
                    ${pyfr.expand('apply_filter_single', 'up', 'fnew', 'd', 'p', 'e')};

                    // Update brackets
                    if (d < ${d_min} || p < ${p_min} || e < entmin - ${e_tol})
                        f_high = fnew;
                    else
                        f_low = fnew;
                }

                // Set current minimum f as the bounds-preserving value
                f = f_low;
            }
        }

        // Filter full solution with bounds-preserving f value
        ${pyfr.expand('apply_filter_full', 'umodes', 'vdm', 'u', 'f')};

        // Calculate minimum entropy from filtered solution
        ${pyfr.expand('get_minima', 'u', 'm0', 'dmin', 'pmin', 'emin')};
        % endif
    }

    // Set new minimum entropy within element for next stage
% for fidx in range(nfaces):
    entmin_int[${fidx}] = emin;
% endfor
</%pyfr:kernel>
