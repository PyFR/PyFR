<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.macros.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.euler.kernels.macros.bcs.${bctype}'/>

## Boundary common normal flux of the system
<%pyfr:macro name='bcflux' params='ul, nl, fn' externs='ploc, t'>
    // Compute the ghost state
    fpdtype_t ur[${nvars}];
    ${pyfr.expand('bc_rsolve_state', 'ul', 'nl', 'ur')};

    // Perform the Riemann solve
    ${pyfr.expand('rsolve', 'ul', 'ur', 'nl', 'fn')};
</%pyfr:macro>
