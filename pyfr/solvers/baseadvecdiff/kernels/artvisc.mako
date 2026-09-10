<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='interp_artvisc' params='V, x, av'>
% if shock_capturing == 'artificial-viscosity':
    av = ${interp_expr};
% else:
    av = 0;
% endif
</%pyfr:macro>

<%pyfr:macro name='artificial_viscosity_add' params='grad_uin, fout, artvisc'>
% if shock_capturing == 'artificial-viscosity':
% for i, j in pyfr.ndrange(ndims, nvars):
    fout[${i}][${j}] -= artvisc*grad_uin[${i}][${j}];
% endfor
% endif
</%pyfr:macro>
