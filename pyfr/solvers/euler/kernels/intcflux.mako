<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<% urstate = 'url' if rperiodic else 'ur' %>
<% rflux = 'fn' if rperiodic else 'ur' %>

<%pyfr:kernel name='intcflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              ur='inout view fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              rmat='in broadcast fpdtype_t[${str(ndims)}][${str(ndims)}]'>
    fpdtype_t mag_nl = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_nl[] = ${pyfr.array('(1 / mag_nl)*nl[{i}]', i=ndims)};

% if rperiodic:
    // Rotate the RHS momentum into the LHS frame: R^T
    fpdtype_t url[] = ${pyfr.carray(['ur[0]', *pyfr.matvec('rmat', 'ur[{j} + 1]', ndims, True), f'ur[{nvars - 1}]'])};
% endif

    // Perform the Riemann solve in the LHS frame
    fpdtype_t fn[${nvars}];
    ${pyfr.expand('rsolve', 'ul', urstate, 'norm_nl', 'fn')};

    // Scale and write out the common normal fluxes
% for i in range(nvars):
    ul[${i}] =  mag_nl*fn[${i}];
    ${rflux}[${i}] = -mag_nl*fn[${i}];
% endfor

% if rperiodic:
    // Rotate the common normal flux into the RHS frame
% for i, expr in enumerate(['fn[0]', *pyfr.matvec('rmat', 'fn[{j} + 1]', ndims), f'fn[{nvars - 1}]']):
    ur[${i}] = ${expr};
% endfor
% endif
</%pyfr:kernel>
