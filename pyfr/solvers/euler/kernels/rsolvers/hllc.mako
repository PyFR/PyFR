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

    // Compute the Roe-averaged enthalpy
    fpdtype_t H = (sqrt(ul[0])*(pr + ur[${ndims + 1}])
                 + sqrt(ur[0])*(pl + ul[${ndims + 1}]))
                / (sqrt(ul[0])*ur[0] + sqrt(ur[0])*ul[0]);

    // Roe average sound speed
    fpdtype_t u = (sqrt(ul[0])*vl[0] + sqrt(ur[0])*vr[0]) /
                  (sqrt(ul[0]) + sqrt(ur[0]));
    fpdtype_t a = sqrt(${c['gamma'] - 1}*(H - 0.5*u*u));

    // Estimate the left and right wave speed, sl and sr
    fpdtype_t sl = u - a;
    fpdtype_t sr = u + a;
    fpdtype_t sstar = (pr - pl + ul[0]*vl[0]*(sl - vl[0])
                               - ur[0]*vr[0]*(sr - vr[0])) /
                      (ul[0]*(sl - vl[0]) - ur[0]*(sr - vr[0]));

    // Star state common factors
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
    usl[${nvars - 1}] = ul_com*(ul[${nvars - 1}] + (sstar - vl[0])*
                                (ul[0]*sstar + pl/(sl - vl[0])));
    usr[${nvars - 1}] = ur_com*(ur[${nvars - 1}] + (sstar - vr[0])*
                                (ur[0]*sstar + pr/(sr - vr[0])));

    // Output
% for i in range(nvars):
    fsl = fl[${i}] + sl*(usl[${i}] - ul[${i}]);
    fsr = fr[${i}] + sr*(usr[${i}] - ur[${i}]);
    nf[${i}] = (0 <= sl) ? fl[${i}] : (sl <= 0 && 0 <= sstar) ? fsl :
               (sstar <= 0 && 0 <= sr) ? fsr : fr[${i}];
% endfor
</%pyfr:macro>

<%include file='pyfr.solvers.euler.kernels.rsolvers.rsolve1d'/>
