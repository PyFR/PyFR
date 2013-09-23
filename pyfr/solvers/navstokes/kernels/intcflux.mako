# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<% beta, tau = c['ldg-beta'], c['ldg-tau'] %>

<%pyfr:kernel name='intcflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              ur='inout view fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              gradur='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              magnl='in fpdtype_t'
              magnr='in fpdtype_t'>
% if beta != -0.5:
    fpdtype_t fvl[${ndims}][${nvars}] = {};
    viscous_flux_add(ul, gradul, fvl);
% endif

% if beta != 0.5:
    fpdtype_t fvr[${ndims}][${nvars}] = {};
    viscous_flux_add(ur, gradur, fvr);
% endif

    // Perform the Riemann solve
    fpdtype_t ficomm[${nvars}], fvcomm;
    rsolve(ul, ur, nl, ficomm);

    for (int i = 0; i < ${nvars}; i++)
    {
% if beta == -0.5:
        fvcomm = ${pyfr.dot('nl[{j}]', 'fvr[{j}][i]', j=ndims)};
% elif beta == 0.5:
        fvcomm = ${pyfr.dot('nl[{j}]', 'fvl[{j}][i]', j=ndims)};
% else:
        fvcomm = ${0.5 + beta}*${pyfr.dot('nl[{j}]', 'fvl[{j}][i]', j=ndims)}
               + ${0.5 - beta}*${pyfr.dot('nl[{j}]', 'fvr[{j}][i]', j=ndims)};
% endif
% if tau != 0.0:
        fvcomm += ${tau}*(ul[i] - ur[i]);
% endif

        ul[i] =  magnl*(ficomm[i] + fvcomm);
        ur[i] = -magnr*(ficomm[i] + fvcomm);
    }
</%pyfr:kernel>
