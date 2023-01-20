<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<%pyfr:macro name='rsolve_1d' params='ul, ur, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${nvars}], fr[${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}];
    fpdtype_t pl, pr, fsl, fsr;
    fpdtype_t usl[${nvars}], usr[${nvars}];

    ${pyfr.expand('inviscid_flux_1d', 'ul', 'fl', 'pl', 'vl')};
    ${pyfr.expand('inviscid_flux_1d', 'ur', 'fr', 'pr', 'vr')};

    // Wave speeds
    fpdtype_t cl = sqrt(${c['gamma']}*pl / ul[0]);
    fpdtype_t cr = sqrt(${c['gamma']}*pr / ur[0]);
    fpdtype_t sl = min(vl[0] - cl, vr[0] - cr);
    fpdtype_t sr = min(vl[0] + cl, vr[0] + cr);
    fpdtype_t sstar = (pr - pl + ul[1]*(sl - vl[0]) - ur[1]*(sr - vr[0])) /
                      (ul[0]*(sl - vl[0]) - ur[0]*(sr - vr[0]));

    fpdtype_t ul_com = (sl - vl[0]) / (sl - sstar);
    fpdtype_t ur_com = (sr - vr[0]) / (sr - sstar);

    // Star state mass
    usl[0] = ul_com*ul[0];
    usr[0] = ur_com*ur[0];

    // Star state momenetum
    usl[1] = ul_com*ul[0]*sstar;
    usr[1] = ur_com*ur[0]*sstar;
% for i in range(2, ndims + 1):
    usl[${i}] = ul_com*ul[${i}];
    usr[${i}] = ur_com*ur[${i}];
% endfor

    // Star state energy
    usl[${nvars - 1}] = ul_com*(ul[${nvars - 1}] + ul[0]*(sstar - vl[0]) *
                                (sstar + pl/(ul[0]*(sl - vl[0]))));
    usr[${nvars - 1}] = ur_com*(ur[${nvars - 1}] + ur[0]*(sstar - vr[0]) *
                                (sstar + pr/(ur[0]*(sr - vr[0]))));

    // Output
% for i in range(nvars):
    fsl = fl[${i}] + sl*(usl[${i}] - ul[${i}]);
    fsr = fr[${i}] + sr*(usr[${i}] - ur[${i}]);
    nf[${i}] = (0 <= sl) ? fl[${i}] : (sl <= 0 && 0 <= sstar) ? fsl : (sstar <= 0 && 0 <= sr) ? fsr : fr[${i}];
% endfor
</%pyfr:macro>

<%include file='pyfr.solvers.euler.kernels.rsolvers.rsolve1d'/>
