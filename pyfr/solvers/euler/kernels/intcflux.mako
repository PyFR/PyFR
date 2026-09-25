<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.periodic'/>
<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<% urstate = 'ur' if rot is None else 'url' %>
<% rflux = 'ur' if rot is None else 'fn' %>

<%pyfr:kernel name='intcflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              ur='inout view fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'>
    fpdtype_t mag_nl = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_nl[] = ${pyfr.array('(1 / mag_nl)*nl[{i}]', i=ndims)};

% if rot is not None:
    // Rotate the RHS momentum into the LHS frame: R^T
    fpdtype_t url[${nvars}];
    ${pyfr.expand('rotate_into_lhs', 'url', 'ur', rot)};
% endif

    // Perform the Riemann solve in the LHS frame
    fpdtype_t fn[${nvars}];
    ${pyfr.expand('rsolve', 'ul', urstate, 'norm_nl', 'fn')};

    // Scale and write out the common normal fluxes
% for i in range(nvars):
    ul[${i}] =  mag_nl*fn[${i}];
    ${rflux}[${i}] = -mag_nl*fn[${i}];
% endfor

% if rot is not None:
    // Rotate the common normal flux into the RHS frame
    ${pyfr.expand('rotate_from_lhs', 'ur', 'fn', rot)};
% endif
</%pyfr:kernel>
