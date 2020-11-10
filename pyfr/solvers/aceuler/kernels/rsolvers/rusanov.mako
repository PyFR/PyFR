# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.aceuler.kernels.flux'/>

<%pyfr:macro name='rsolve_t1d' params='ul, ur, nf'>
    fpdtype_t fl[${nvars}], fr[${nvars}];

    ${pyfr.expand('inviscid_1dflux', 'ul', 'fl')};
    ${pyfr.expand('inviscid_1dflux', 'ur', 'fr')};

    // Quasi-Davis max wavespeed
    fpdtype_t a = max(fabs(ul[1] + sqrt(ul[1]*ul[1] + ${c['ac-zeta']})),
                      fabs(ur[1] + sqrt(ur[1]*ur[1] + ${c['ac-zeta']})));

    // Output
% for i in range(nvars):
    nf[${i}] = 0.5*(fl[${i}] + fr[${i}]) + 0.5*a*(ul[${i}] - ur[${i}]);
% endfor
</%pyfr:macro>

<%include file='pyfr.solvers.aceuler.kernels.rsolvers.rsolve_trans'/>
