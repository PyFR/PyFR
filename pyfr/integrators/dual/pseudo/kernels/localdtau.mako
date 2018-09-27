# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='localdtau' ndim='2'
              negdivconf='inout fpdtype_t[${str(nvars)}]'
              dtau_upts='in fpdtype_t[${str(nvars)}]'
              inv='scalar fpdtype_t'>
% for i in range(nvars):
    negdivconf[${i}] = ((1.0 - inv)*dtau_upts[${i}]
                        + inv/dtau_upts[${i}])*negdivconf[${i}];
% endfor
</%pyfr:kernel>
