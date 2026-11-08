<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.navstokes.kernels.macros.bcs.${bctype}'/>

% if bccfluxstate:
<%include file='pyfr.solvers.navstokes.kernels.macros.bcs.${bccfluxstate}'/>
% endif

<%pyfr:macro name='bcflux' params='ul, gradul, artvisc, nl, magnl, fn'
             externs='ploc, t'>
% for i in range(nvars):
    fn[${i}] = ul[${i}];
% endfor
    ${pyfr.expand('bc_common_flux_state', 'fn', 'gradul', 'artvisc', 'nl',
                  'magnl')};
</%pyfr:macro>
