<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.entropy'/>

<%pyfr:kernel name='entropylocal' ndim='1'
              u='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              entmin_int='out fpdtype_t[${str(nfaces)}]'
              m0='in broadcast fpdtype_t[${str(nfpts)}][${str(nupts)}]'>
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

    % if not fpts_in_upts:
    fpdtype_t uf[${nvars}];
    for (int fidx = 0; fidx < ${nfpts}; fidx++)
    {
        % for vidx in range(nvars):
        uf[${vidx}] = ${pyfr.dot('m0[fidx][{k}]', f'u[{{k}}][{vidx}]', k=nupts)};
        % endfor
        ${pyfr.expand('compute_entropy', 'uf', 'd', 'p', 'e')};
        entmin = fmin(entmin, e);
    }
    % endif

    // Set interface entropy values to minimum
% for i in range(nfaces):
    entmin_int[${i}] = entmin;
% endfor
</%pyfr:kernel>
