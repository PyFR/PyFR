<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.macros.inviscid'/>
<%include file='pyfr.solvers.navstokes.kernels.macros.viscous'/>

<%pyfr:macro name='flux' params='uin, grad_uin, fout'>
    fpdtype_t p, v[${ndims}];
    ${pyfr.expand('inviscid_flux', 'uin', 'fout', 'p', 'v')};
    ${pyfr.expand('viscous_flux_add', 'uin', 'grad_uin', 'fout')};
</%pyfr:macro>
