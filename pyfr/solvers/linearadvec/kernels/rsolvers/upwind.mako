# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.linearadvec.kernels.flux'/>

<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];

    ${pyfr.expand('inviscid_flux', 'ul', 'fl')};
    ${pyfr.expand('inviscid_flux', 'ur', 'fr')};

    // Output
% for i in range(nvars):
    nf[${i}] = ${" + ".join(f"(n[{j}]*{c['a'][j]} > 0 ? n[{j}]*fl[{j}][{i}] : n[{j}]*fr[{j}][{i}])" for j in range(ndims))};
% endfor
</%pyfr:macro>
