# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='gradcoru' ndim='2'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'>
    fpdtype_t tmpgradu[${ndims}];

% for j in range(nvars):
% for i in range(ndims):
    tmpgradu[${i}] = gradu[${i}][${j}];
% endfor
% for i in range(ndims):
    gradu[${i}][${j}] = rcpdjac*(${' + '.join('smats[{k}][{i}]*tmpgradu[{k}]'
                                              .format(i=i, k=k)
                                              for k in range(ndims))});
% endfor
% endfor
</%pyfr:kernel>
