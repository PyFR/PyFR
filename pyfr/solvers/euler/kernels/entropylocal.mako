# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.entropy'/>

<% inf = 1e20%>
<%pyfr:kernel name='entropylocal' ndim='1'
              u='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              entmin='out fpdtype_t'
              entmin_int='out fpdtype_t[${str(nfpts)}]'>

    // Compute minimum entropy across element
    fpdtype_t ui[${nvars}], d, p, e;

    entmin = ${inf};
    fpdtype_t entmax = -${inf};

% for i in range(nupts):
    % for j in range(nvars):
    ui[${j}] = u[${i}][${j}];
    % endfor
    
    ${pyfr.expand('compute_entropy', 'ui', 'd', 'p', 'e')};
    
    entmin = fmin(entmin, e);
    entmax = fmax(entmax, e);
% endfor

    // Compute relative undershoot tolerance
    entmin -= ${e_rtol}*(entmax - entmin);

    // Set interface entropy values to minimum
% for i in range(nfpts):
    entmin_int[${i}] = entmin;
% endfor

</%pyfr:kernel>
