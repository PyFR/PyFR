<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='compute_entropy' params='u, d, p, e'>
    d = u[0];
    fpdtype_t rcpd = 1.0/d;
    fpdtype_t E = u[${nvars - 1}];

    // Compute the pressure
    p = ${c['gamma'] - 1}*(E - 0.5*rcpd*(${pyfr.dot('u[{i}]', i=(1, ndims + 1))}));

    // Compute entropy
    // Testing showed the functional s=exp(log(p*r^-g))
    // yielded more consistent results across a range
    // of pressure and density magnitudes, as well as being less sensitive
    // to small purturbations in smooth regions of the flow.

    e = (d > 0 && p > 0) ? p*pow(rcpd, ${c['gamma']}) : ${fpdtype_max};
</%pyfr:macro>
