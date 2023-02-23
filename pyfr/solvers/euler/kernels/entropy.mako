<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% inf = 1e20 %>
<%pyfr:macro name='compute_entropy' params='u, d, p, e'>
    d = u[0];
    fpdtype_t rcpd = 1.0/d;
    fpdtype_t E = u[${nvars - 1}];

    // Compute the pressure
    p = ${c['gamma'] - 1}*(E - 0.5*rcpd*(${pyfr.dot('u[{i}+1]', i=ndims)}));

    // Compute specific physical entropy
    e = ((d > 0) && (p > 0)) ? p*pow(rcpd, ${c['gamma']}) : ${inf};
</%pyfr:macro>
