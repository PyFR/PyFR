# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% inf = 1e20%>
<%pyfr:macro name='compute_entropy' params='u, d, p, e'>
    d = u[0];
    fpdtype_t E = u[${nvars - 1}];

    // Compute the velocities
    fpdtype_t rhov[${ndims}];
% for i in range(ndims):
    rhov[${i}] = u[${i + 1}];
% endfor

    // Compute the pressure
    p = ${c['gamma'] - 1}*(E - 0.5*(${pyfr.dot('rhov[{i}]', i=ndims)})/d);

    // Compute specific physical entropy
    e = ((d > 0) && (p > 0)) ? p*pow(d, ${-c['gamma']}) : ${inf};
</%pyfr:macro>