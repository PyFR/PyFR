# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.acnavstokes.kernels.bcs.${bctype}'/>

<%pyfr:kernel name='bcconu' ndim='1'
              ulin='in view fpdtype_t[${str(nvars)}]'
              ulout='out view fpdtype_t[${str(nvars)}]'
              nlin='in fpdtype_t[${str(ndims)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              t='scalar fpdtype_t'>
    ${pyfr.expand('bc_ldg_state', 'ulin', 'nlin', 'ulout', 'ploc', 't')};
</%pyfr:kernel>
