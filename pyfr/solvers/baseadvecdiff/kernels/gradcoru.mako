# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='gradcoru' ndim='2'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'>
    for (int j = 0; j < ${nvars}; j++)
    {
        fpdtype_t tmpgradu[${ndims}];
    % for i in range(ndims):
        tmpgradu[${i}] = gradu[${i}][j];
    % endfor

    % for i in range(ndims):
        gradu[${i}][j] = rcpdjac*(${' + '.join('smats[{0}][{1}]*tmpgradu[{0}]'
                                               .format(k, i)
                                               for k in range(ndims))});
    % endfor
    }
</%pyfr:kernel>
