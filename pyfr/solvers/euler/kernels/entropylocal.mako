<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.entropy'/>

<%pyfr:kernel name='entropylocal' ndim='1'
              u='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              entmin_int='out fpdtype_t[${str(nfaces)}]'>
    // Compute minimum entropy across element
    fpdtype_t ui[${nvars}], d, p, e;

    fpdtype_t entmin = ${fpdtype_max};
    for (int i = 0; i < ${nupts}; i++)
    {
    % for j in range(nvars):
        ui[${j}] = u[i][${j}];
    % endfor

        ${pyfr.expand('compute_entropy', 'ui', 'd', 'p', 'e')};

        entmin = fmin(entmin, e);
    }

    // Set interface entropy values to minimum
% for i in range(nfaces):
    entmin_int[${i}] = entmin;
% endfor
</%pyfr:kernel>
