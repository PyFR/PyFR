<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.macros.inviscid'/>

<%pyfr:macro name='flux' params='uin, fout'>
    fpdtype_t p, v[${ndims}];
    ${pyfr.expand('inviscid_flux', 'uin', 'fout', 'p', 'v')};
</%pyfr:macro>
