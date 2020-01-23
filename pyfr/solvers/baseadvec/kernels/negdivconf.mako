# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='negdivconf' ndim='2'
              t='scalar fpdtype_t'
              tdivtconf='inout fpdtype_t[${str(nvars)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              u='in fpdtype_t[${str(nvars)}]'
              rcpdjac='in fpdtype_t'>
% for i, ex in enumerate(srcex):
    tdivtconf[${i}] = -rcpdjac*tdivtconf[${i}] + ${ex};
% endfor

// source term for synthetic turbulence, only for the momentum equations
% for i in range(ndims):
    tdivtconf[${i} + 1] += turbsrc[${i}]
% endfor

</%pyfr:kernel>
