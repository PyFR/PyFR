<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.navstokes.kernels.bcs.${bctype}'/>

<%pyfr:kernel name='bccent' ndim='1'
              ul='in view fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              entmin_lhs='out view reduce(min) fpdtype_t'>
    fpdtype_t mag_nl = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_nl[] = ${pyfr.array('(1 / mag_nl)*nl[{i}]', i=ndims)};

    // Compute the RHS
    fpdtype_t ur[${nvars}];
    ${pyfr.expand('bc_rsolve_state', 'ul', 'norm_nl', 'ur')};

    // Compute entropy for the boundary state (reduce with atomics)
    ${fluid.decl('ur', 's', suffix='_ce')}
    entmin_lhs = s_ce;
</%pyfr:kernel>
