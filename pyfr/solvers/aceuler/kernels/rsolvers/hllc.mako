# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.aceuler.kernels.flux'/>

<% zeta = c['ac-zeta'] %>
<% rzeta = 1/c['ac-zeta'] %>

<%pyfr:macro name='rsolve_t1d' params='ul, ur, nf'>
    fpdtype_t fl[${nvars}], fr[${nvars}];
    fpdtype_t fsl, fsr;
    fpdtype_t usl[${nvars}], usr[${nvars}];

    // Compute the left and right fluxes
    ${pyfr.expand('inviscid_1dflux', 'ul', 'fl')};
    ${pyfr.expand('inviscid_1dflux', 'ur', 'fr')};
    
    // Estimate the left and right wave speed, sl and sr
    fpdtype_t ua = 0.5*(ul[1] + ur[1]);
    fpdtype_t aa = sqrt(ua*ua + ${zeta});
    fpdtype_t sl = ua - aa;
    fpdtype_t sr = ua + aa;

    // HLLC Star region values
    fpdtype_t inv_ds = 1/(sr - sl);
    fpdtype_t ps = (sr*ur[0] - sl*ul[0] + ${zeta}*(ul[1] - ur[1]))*inv_ds;
    fpdtype_t us = (${rzeta}*sl*sr*(ur[0] - ul[0]) + (ul[1]*sr - ur[1]*sl))*inv_ds;

    fpdtype_t rsl = 1 / (sl - us);
    fpdtype_t rsr = 1 / (sr - us);
    usl[0] = ps;
    usl[1] = us;

    usr[0] = ps;
    usr[1] = us;
% for i in range(2,ndims + 1):
    usl[${i}] = ul[${i}]*(sl - ul[1])*rsl;
    usr[${i}] = ur[${i}]*(sr - ur[1])*rsr;
% endfor

    // Output
% for i in range(nvars):
    fsl = fl[${i}] + sl*(usl[${i}] - ul[${i}]);
    fsr = fr[${i}] + sr*(usr[${i}] - ur[${i}]);
    
    nf[${i}] = (sl >= 0) ? fl[${i}] : (sl <= 0 && us >= 0) ? fsl :
               (us <= 0 && sr >= 0) ? fsr : fr[${i}];
% endfor
</%pyfr:macro>

<%include file='pyfr.solvers.aceuler.kernels.rsolvers.rsolve_trans'/>
