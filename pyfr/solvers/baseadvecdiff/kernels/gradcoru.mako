# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='gradcoru' ndim='2'
              jmats='in fpdtype_t[${str(ndims**2)}]'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'>
    for (int j = 0; j < ${nvars}; j++)
    {
        fpdtype_t tmpgradu[${ndims}];
    % for i in range(ndims):
        tmpgradu[${i}] = gradu[${i}][j];
    % endfor

    % for i in range(ndims):
        gradu[${i}][j] = ${' + '.join('jmats[{0}]*tmpgradu[{1}]'
                                      .format(i*ndims + k, k)
                                      for k in range(ndims))};
    % endfor
    }
</%pyfr:kernel>
