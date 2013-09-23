# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:function name='bc_common_grad_zero'
                params='fpdtype_t ul[${str(nvars)}],
                        fpdtype_t nl[${str(ndims)}],
                        fpdtype_t gradul[${str(ndims)}][${str(nvars)}],
                        fpdtype_t gradur[${str(ndims)}][${str(nvars)}]'>
    for (int i = 0; i < ${ndims}; i++)
        for (int j = 0; j < ${nvars}; j++)
            gradur[i][j] = 0;
</%pyfr:function>

<%pyfr:function name='bc_common_grad_copy'
                params='fpdtype_t ul[${str(nvars)}],
                        fpdtype_t nl[${str(ndims)}],
                        fpdtype_t gradul[${str(ndims)}][${str(nvars)}],
                        fpdtype_t gradur[${str(ndims)}][${str(nvars)}]'>
    for (int i = 0; i < ${ndims}; i++)
        for (int j = 0; j < ${nvars}; j++)
            gradur[i][j] = gradul[i][j];
</%pyfr:function>
