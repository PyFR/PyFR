# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='bc_common_grad_zero' params='ul, nl, gradul, gradur'>
% for i, j in pyfr.ndrange(ndims, nvars):
    gradur[${i}][${j}] = 0;
% endfor
</%pyfr:macro>

<%pyfr:macro name='bc_common_grad_copy' params='ul, nl, gradul, gradur'>
% for i, j in pyfr.ndrange(ndims, nvars):
    gradur[${i}][${j}] = gradul[${i}][${j}];
% endfor
</%pyfr:macro>
