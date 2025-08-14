<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}];
    fpdtype_t pl, pr, nf_fl, nf_fr, nf_fsl, nf_fsr;
    fpdtype_t va[${ndims}];
    fpdtype_t usl[${nvars}], usr[${nvars}];
    fpdtype_t sqrtrl = sqrt(ul[0]);
    fpdtype_t sqrtrr = sqrt(ur[0]);

    ${pyfr.expand('inviscid_flux', 'ul', 'fl', 'pl', 'vl')};
    ${pyfr.expand('inviscid_flux', 'ur', 'fr', 'pr', 'vr')};

    // Get the normal left and right velocities
    fpdtype_t nvl = ${pyfr.dot('n[{i}]', 'vl[{i}]', i=ndims)};
    fpdtype_t nvr = ${pyfr.dot('n[{i}]', 'vr[{i}]', i=ndims)};

    fpdtype_t al = sqrt(${c['gamma']}*pl/ul[0]);
    fpdtype_t ar = sqrt(${c['gamma']}*pr/ur[0]);

    // Compute the Roe-averaged velocity
    fpdtype_t nv = (sqrtrl*nvl + sqrtrr*nvr)
                 / (sqrtrl + sqrtrr);

    // Compute the Roe-averaged enthalpy
    fpdtype_t H = (sqrtrl*(pr + ur[${ndims + 1}])
                 + sqrtrr*(pl + ul[${ndims + 1}]))
                / (sqrtrl*ur[0] + sqrtrr*ul[0]);

    fpdtype_t inv_rar = 1 / (sqrtrl + sqrtrr);
% for i in range(ndims):
    va[${i}] = (vl[${i}]*sqrtrl + vr[${i}]*sqrtrr) * inv_rar;
% endfor

    fpdtype_t qq = ${pyfr.dot('va[{i}]', i=ndims)};

    // Roe average speed of sound
    fpdtype_t a = sqrt(${c['gamma'] - 1}*(H - 0.5*qq));

    // Estimate the left and right wave speed, sl and sr
    fpdtype_t sl = min(nv - a, nvl - al);
    fpdtype_t sr = max(nv + a, nvr + ar);
    fpdtype_t sstar = (pr - pl + ul[0]*nvl*(sl - nvl)
                               - ur[0]*nvr*(sr - nvr)) /
                      (ul[0]*(sl - nvl) - ur[0]*(sr - nvr));

    // Star state common factors
    fpdtype_t ul_com = (sl - nvl) / (sl - sstar);
    fpdtype_t ur_com = (sr - nvr) / (sr - sstar);

    // Star state mass
    usl[0] = ul_com*ul[0];
    usr[0] = ur_com*ur[0];

    // Star state momenetum
% for i in range(ndims):
    usl[${i + 1}] = usl[0]*(vl[${i}] + (sstar - nvl)*n[${i}]);
    usr[${i + 1}] = usr[0]*(vr[${i}] + (sstar - nvr)*n[${i}]);
%endfor

    // Star state energy
    usl[${nvars - 1}] = ul_com*(ul[${nvars - 1}] + (sstar - nvl)*
                                (ul[0]*sstar + pl/(sl - nvl)));
    usr[${nvars - 1}] = ur_com*(ur[${nvars - 1}] + (sstar - nvr)*
                                (ur[0]*sstar + pr/(sr - nvr)));

    // Output
% for i in range(nvars):
    nf_fl = ${' + '.join(f'n[{j}]*fl[{j}][{i}]' for j in range(ndims))};
    nf_fr = ${' + '.join(f'n[{j}]*fr[{j}][{i}]' for j in range(ndims))};
    nf_fsl = nf_fl + sl*(usl[${i}] - ul[${i}]);
    nf_fsr = nf_fr + sr*(usr[${i}] - ur[${i}]);
    nf[${i}] = (0 <= sl) ? nf_fl : (sl <= 0 && 0 <= sstar) ? nf_fsl :
               (sstar <= 0 && 0 <= sr) ? nf_fsr : nf_fr;
% endfor
</%pyfr:macro>
