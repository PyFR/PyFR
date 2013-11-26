# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='gradcoru' ndim='2'
              smats='in fpdtype_t[${str(ndims**2)}]'
              rcpdjac='in fpdtype_t'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'>
    for (int j = 0; j < ${nvars}; j++)
    {
        fpdtype_t tmpgradu[${ndims}];
    % for i in range(ndims):
        tmpgradu[${i}] = gradu[${i}][j];
    % endfor

    % for i in range(ndims):
        gradu[${i}][j] = rcpdjac*(${' + '.join('smats[{0}]*tmpgradu[{1}]'
                                               .format(k*ndims + i, k)
                                               for k in range(ndims))});
    % endfor
    }
</%pyfr:kernel>
