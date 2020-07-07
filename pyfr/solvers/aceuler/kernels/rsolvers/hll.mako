# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.aceuler.kernels.flux1d'/>

<%pyfr:macro name='rsolve_t1d' params='ul, ur, nf'>
    fpdtype_t fl[${nvars}], fr[${nvars}];
    fpdtype_t nf_sub;

    // Compute the left and right fluxes
    ${pyfr.expand('inviscid_1dflux', 'ul', 'fl')};
    ${pyfr.expand('inviscid_1dflux', 'ur', 'fr')};

    // Estimate the left and right wave speed, sl and sr
    fpdtype_t sl = ul[1] - sqrt(ul[1]*ul[1] + ${c['ac-zeta']});
    fpdtype_t sr = ur[1] + sqrt(ur[1]*ur[1] + ${c['ac-zeta']});
    fpdtype_t rcpsrsl = 1./(sr - sl);

    // Output
% for i in range(nvars):
    nf_sub = (sr*fl[${i}] - sl*fr[${i}] + sl*sr*(ur[${i}] - ul[${i}]))*rcpsrsl;
    
    nf[${i}] = (0. <= sl) ? fl[${i}] : (0. >= sr) ? fr[${i}] : nf_sub;
% endfor
</%pyfr:macro>

<%include file='pyfr.solvers.aceuler.kernels.rsolvers.rsolve_trans'/>