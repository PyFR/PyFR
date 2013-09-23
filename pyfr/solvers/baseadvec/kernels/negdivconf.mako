# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='negdivconf' ndim='2'
              tdivtconf='inout fpdtype_t[${str(nvars)}]'
              rcpdjac='in fpdtype_t'>
% for i in range(nvars):
    tdivtconf[${i}] *= -rcpdjac;
% endfor
</%pyfr:kernel>
