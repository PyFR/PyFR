# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.entropy'/>

<% inf = 1e20 %>
<%pyfr:kernel name='entropylocal' ndim='1'
              u='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              entmin_int='out fpdtype_t[${str(nfpts)}]'>
    // Compute minimum entropy across element
    fpdtype_t ui[${nvars}], d, p, e;

    fpdtype_t entmin = ${inf};
    for (int i = 0; i < ${nupts}; i++)
    {
        % for j in range(nvars):
        ui[${j}] = u[i][${j}];
        % endfor

        ${pyfr.expand('compute_entropy', 'ui', 'd', 'p', 'e')};

        entmin = fmin(entmin, e);
    }

    // Set interface entropy values to minimum
    for (int i = 0; i < ${nfpts}; i++)
    {
        entmin_int[i] = entmin;
    }

</%pyfr:kernel>
