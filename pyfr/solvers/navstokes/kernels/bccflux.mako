# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.navstokes.kernels.bcs.${bctype}'/>

% if bccfluxstate:
<%include file='pyfr.solvers.navstokes.kernels.bcs.${bccfluxstate}'/>
% endif

% if stv:
${'\n'.join("#define {} {}".format(ci, cv) for ci, cv in constants.items())}

<%pyfr:kernel name='bccflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              amul='in view fpdtype_t'
              nl='in fpdtype_t[${str(ndims)}]'
              magnl='in fpdtype_t'
              ploc='in fpdtype_t[${str(ndims)}]'
              t='scalar fpdtype_t'>
    ${pyfr.expand('bc_common_flux_state', 'ul', 'gradul', 'amul', 'nl', 'magnl', 'ploc', 't')};
</%pyfr:kernel>
% else:
<%pyfr:kernel name='bccflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              amul='in view fpdtype_t'
              nl='in fpdtype_t[${str(ndims)}]'
              magnl='in fpdtype_t'>
    ${pyfr.expand('bc_common_flux_state', 'ul', 'gradul', 'amul', 'nl', 'magnl')};
</%pyfr:kernel>
% endif
