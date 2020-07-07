# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='inviscid_1dflux' params='s, f'>
    // Pressure in the conservative variable array index 0
    fpdtype_t p = s[0];

    // Mass flux
    f[0] = ${c['ac-zeta']}*s[1];
    
    // Momentum fluxes
% for i in range(ndims):
    f[${i+1}] = s[1]*s[${i+1}]${' + p' if i == 0 else ''};
% endfor
</%pyfr:macro>