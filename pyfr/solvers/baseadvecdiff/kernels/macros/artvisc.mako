<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='interp_artvisc' params='V, avop, av'>
% if shock_capturing == 'artificial-viscosity':
    av = ${' + '.join(f'avop[{i}]*V[{i}]' for i in range(nverts))};
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

<%pyfr:macro name='artificial_viscosity_add_wall' params='grad_uin, fout, artvisc'>
% if shock_capturing == 'artificial-viscosity':
% for i, j in pyfr.ndrange(ndims, nvars):
% if 0 < j < nvars - 1:
    fout[${i}][${j}] -= artvisc*grad_uin[${i}][${j}];
% endif
% endfor
% endif
</%pyfr:macro>
