# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.linearadvec.kernels.flux'/>

<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];

    ${pyfr.expand('inviscid_flux', 'ul', 'fl')};
    ${pyfr.expand('inviscid_flux', 'ur', 'fr')};

    fpdtype_t sgn[${ndims}];
% for i in range(ndims):
	sgn[${i}] = n[${i}]*${c['a'][i]} > 0 ? 1 : -1;
% endfor

    // Output
% for i in range(nvars):
    nf[${i}] = ${" + ".join(f'n[{j}]*((0.5 + sgn[{j}]*{alpha/2})*fl[{j}][{i}]'
    	                    f' + (0.5 - sgn[{j}]*{alpha/2})*fr[{j}][{i}])' 
    	                    for j in range(ndims))};
% endfor
</%pyfr:macro>