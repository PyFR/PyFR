# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='gradcoru' ndim='2'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'>
    fpdtype_t tmpgradu[${ndims}];

% for j in range(nvars):
% for i in range(ndims):
    tmpgradu[${i}] = gradu[${i}][${j}];
% endfor
% for i in range(ndims):
    gradu[${i}][${j}] = rcpdjac*(${' + '.join(f'smats[{k}][{i}]*tmpgradu[{k}]'
                                              for k in range(ndims))});
% endfor
% endfor
</%pyfr:kernel>
