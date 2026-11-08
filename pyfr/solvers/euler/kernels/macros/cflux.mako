<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.macros.rsolvers.${rsolver}'/>

<%pyfr:macro name='cflux' params='ul, ur, n, fn'>
    // Perform the Riemann solve
    ${pyfr.expand('rsolve', 'ul', 'ur', 'n', 'fn')};
</%pyfr:macro>
