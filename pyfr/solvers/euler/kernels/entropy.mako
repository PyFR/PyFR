<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='compute_entropy' params='u, d, p, e'>
    ${fluid.decl('u', 'rho, p, s', suffix='_ce')}
    d = rho_ce;
    p = p_ce;
    e = s_ce;
</%pyfr:macro>
