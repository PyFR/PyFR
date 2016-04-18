# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='shockvar' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              s='out fpdtype_t'>
    s = u[${shockvar}];
</%pyfr:kernel>
