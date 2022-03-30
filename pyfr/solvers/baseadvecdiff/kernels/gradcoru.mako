# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='gradcoru' ndim='2'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'>
    fpdtype_t tmpgradu[${ndims}][${nvars}];

% for i, j in pyfr.ndrange(ndims, nvars):
    tmpgradu[${i}][${j}] = gradu[${i}][${j}];
% endfor

% for i, j in pyfr.ndrange(ndims, nvars):
    gradu[${i}][${j}] = rcpdjac*(${' + '.join(f'smats[{k}][{i}]*tmpgradu[{k}][{j}]'
                                              for k in range(ndims))});
% endfor
</%pyfr:kernel>
