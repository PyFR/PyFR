# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='artificial_viscosity_add' params='grad_uin, fout, artvisc'>
% if shock_capturing == 'artificial-viscosity':
% for i, j in pyfr.ndrange(ndims, nvars):
    fout[${i}][${j}] -= artvisc*grad_uin[${i}][${j}];
% endfor
% endif
</%pyfr:macro>
