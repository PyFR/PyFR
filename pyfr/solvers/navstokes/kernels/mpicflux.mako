# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<% beta, tau = c['ldg-beta'], c['ldg-tau'] %>

<%pyfr:kernel name='mpicflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              ur='inout mpi fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              gradur='in mpi fpdtype_t[${str(ndims)}][${str(nvars)}]'
              amul='in view fpdtype_t'
              amur='in mpi fpdtype_t'
              spngmu='in fpdtype_t'
              nl='in fpdtype_t[${str(ndims)}]'
              magnl='in fpdtype_t'>
    // Perform the Riemann solve
    fpdtype_t ficomm[${nvars}], fvcomm;
    ${pyfr.expand('rsolve', 'ul', 'ur', 'nl', 'ficomm')};

% if beta != -0.5:
    fpdtype_t fvl[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ul', 'gradul', 'amul', 'spngmu', 'fvl')};
% endif

% if beta != 0.5:
    fpdtype_t fvr[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ur', 'gradur', 'amur', 'spngmu', 'fvr')};
% endif

% for i in range(nvars):
% if beta == -0.5:
    fvcomm = ${' + '.join('nl[{j}]*fvr[{j}][{i}]'.format(i=i, j=j)
                          for j in range(ndims))};
% elif beta == 0.5:
    fvcomm = ${' + '.join('nl[{j}]*fvl[{j}][{i}]'.format(i=i, j=j)
                          for j in range(ndims))};
% else:
    fvcomm = ${0.5 + beta}*(${' + '.join('nl[{j}]*fvl[{j}][{i}]'
                                         .format(i=i, j=j)
                                         for j in range(ndims))})
           + ${0.5 - beta}*(${' + '.join('nl[{j}]*fvr[{j}][{i}]'
                                         .format(i=i, j=j)
                                         for j in range(ndims))});
% endif
% if tau != 0.0:
    fvcomm += ${tau}*(ul[${i}] - ur[${i}]);
% endif

    ul[${i}] =  magnl*(ficomm[${i}] + fvcomm);
% endfor
</%pyfr:kernel>
