# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.bcs.${bctype}'/>
<%include file='pyfr.solvers.euler.kernels.entropy'/>

<%pyfr:kernel name='bccent' ndim='1'
              ul='in view fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              entmin_lhs='out view reduce(min_pos) fpdtype_t'>
    fpdtype_t mag_nl = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_nl[] = ${pyfr.array('(1 / mag_nl)*nl[{i}]', i=ndims)};

    // Compute the RHS
    fpdtype_t ur[${nvars}];
    ${pyfr.expand('bc_rsolve_state', 'ul', 'norm_nl', 'ur')};

    // Compute entropy for boundary state
    fpdtype_t p, d, entmin_rhs;
    ${pyfr.expand('compute_entropy', 'ur', 'd', 'p', 'entmin_rhs')};

    // Compute face minima (reduce with atomics)
    entmin_lhs = entmin_rhs;
</%pyfr:kernel>
