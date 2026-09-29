<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<% urstate = 'ur' if rot is None else 'url' %>
<% rflux = 'ur' if rot is None else 'fn' %>
<% vmap = (None, None, 1) %>

<%pyfr:kernel name='intcflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              ur='inout view fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'>
    fpdtype_t mag_nl = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_nl[] = ${pyfr.array('(1 / mag_nl)*nl[{i}]', i=ndims)};

% if rot is not None:
    // Rotate the RHS momentum into the LHS frame: R^T
    fpdtype_t url[${nvars}];
    url[0] = ur[0];
    url[${nvars - 1}] = ur[${nvars - 1}];
    ${pyfr.constmatvec(rot.T, 'url', 'ur', vmap, vmap)}
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
    ur[0] = fn[0];
    ur[${nvars - 1}] = fn[${nvars - 1}];
    ${pyfr.constmatvec(rot, 'ur', 'fn', vmap, vmap)}
% endif
</%pyfr:kernel>
