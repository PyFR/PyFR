# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.${bctype}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<% tau = c['ldg-tau'] %>

<%pyfr:kernel name='bccflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              magnl='in fpdtype_t'>
    // Viscous states
    fpdtype_t ur[${nvars}], gradur[${ndims}][${nvars}];
    bc_ldg_state(ul, ur);
    bc_ldg_grad_state(ul, nl, gradul, gradur);

    fpdtype_t fvr[${ndims}][${nvars}] = {};
    viscous_flux_add(ur, gradur, fvr);

    // Inviscid (Riemann solve) state
    bc_rsolve_state(ul, ur);

    // Perform the Riemann solve
    fpdtype_t ficomm[${nvars}], fvcomm;
    rsolve(ul, ur, nl, ficomm);

    for (int i = 0; i < ${nvars}; i++)
    {
        fvcomm = ${pyfr.dot('nl[{j}]', 'fvr[{j}][i]', j=ndims)};
% if tau != 0.0:
        fvcomm += ${tau}*(ul[i] - ur[i]);
% endif

        ul[i] = magnl*(ficomm[i] + fvcomm);
    }
</%pyfr:kernel>
