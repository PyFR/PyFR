# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.navstokes.kernels.bcs.${bctype}'/>
<%include file='pyfr.solvers.euler.kernels.entropy'/>

<%pyfr:kernel name='bccent' ndim='1'
              entmin_lhs='inout view fpdtype_t'
              ul='in view fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              magnl='in fpdtype_t'>
    // Compute the RHS
    fpdtype_t ur[${nvars}];
    ${pyfr.expand('bc_rsolve_state', 'ul', 'nl', 'ur')};

    // Compute entropy for boundary state
    fpdtype_t p, d, entmin_rhs;
    ${pyfr.expand('compute_entropy', 'ur', 'd', 'p', 'entmin_rhs')};

    entmin_lhs = fmin(entmin_lhs, entmin_rhs);
</%pyfr:kernel>
