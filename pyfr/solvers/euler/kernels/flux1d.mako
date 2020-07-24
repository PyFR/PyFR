# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='inviscid_1dflux' params='s, f, p, v'>
    fpdtype_t invrho = 1.0/s[0], E = s[${nvars - 1}];
    
    // Compute the velocities
    fpdtype_t rhov[${ndims}];
% for i in range(ndims):
    rhov[${i}] = s[${i + 1}];
    v[${i}] = invrho*rhov[${i}];
% endfor
    // Compute the pressure
    p = ${c['gamma'] - 1}*(E - 0.5*invrho*${pyfr.dot('rhov[{i}]', i=ndims)});
    
    // Density and energy fluxes
    f[0] = rhov[0];
    f[${nvars - 1}] = (E + p)*v[0];
    
    // Momentum fluxes
% for i in range(ndims):
    f[${i + 1}] = rhov[0]*v[${i}]${' + p' if i == 0 else ''};
% endfor
</%pyfr:macro>