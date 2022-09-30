# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>

<%pyfr:kernel name='mpicflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              ur='in mpi fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'>
    fpdtype_t mag_nl = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_nl[] = ${pyfr.array('(1 / mag_nl)*nl[{i}]', i=ndims)};

    // Perform the Riemann solve
    fpdtype_t fn[${nvars}];
    ${pyfr.expand('rsolve', 'ul', 'ur', 'norm_nl', 'fn')};

    // Scale and write out the common normal fluxes
% for i in range(nvars):
    ul[${i}] = mag_nl*fn[${i}];
% endfor
</%pyfr:kernel>
