# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='viscous_flux_add' params='uin, grad_uin, fout'>
% for i, j in pyfr.ndrange(ndims, ndims):
    fout[${i}][${j+1}] += -${c['nu']}*grad_uin[${i}][${j+1}];
% endfor
</%pyfr:macro>
