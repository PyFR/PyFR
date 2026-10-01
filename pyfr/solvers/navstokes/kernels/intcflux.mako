<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.baseadvec.kernels.transform'/>
<%include file='pyfr.solvers.baseadvecdiff.kernels.artvisc'/>
<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<%
beta, tau = c['ldg-beta'], c['ldg-tau']
urstate = 'url' if rperiodic else 'ur'
gradurstate = 'gradurl' if rperiodic else 'gradur'
%>

<%pyfr:kernel name='intcflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              ur='inout view fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              gradur='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              artvisc='in view fpdtype_t'
              nl='in fpdtype_t[${str(ndims)}]'
              rmat='in broadcast fpdtype_t[${str(ndims)}][${str(ndims)}]'>
    fpdtype_t mag_nl = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_nl[] = ${pyfr.array('(1 / mag_nl)*nl[{i}]', i=ndims)};

% if rperiodic:
    // Rotate the RHS momentum into the LHS frame: R^T
    fpdtype_t url[] = ${pyfr.array('ur[{i}]', i=nvars)};
    ${pyfr.expand('rotate', 'rmat', 'url', off=1, transpose=True)};
% endif

% if rperiodic and beta != 0.5:
    // Rotate the RHS gradient into the LHS frame
    fpdtype_t gradurl[${ndims}][${nvars}];
    ${pyfr.expand('rotate_grad', 'rmat', 'gradur', 'gradurl')};
% endif

    // Perform the Riemann solve
    fpdtype_t ficomm[${nvars}], fvcomm;
    ${pyfr.expand('rsolve', 'ul', urstate, 'norm_nl', 'ficomm')};

% if beta != -0.5:
    fpdtype_t fvl[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ul', 'gradul', 'fvl')};
    ${pyfr.expand('artificial_viscosity_add', 'gradul', 'fvl', 'artvisc')};
% endif

% if beta != 0.5:
    fpdtype_t fvr[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', urstate, gradurstate, 'fvr')};
    ${pyfr.expand('artificial_viscosity_add', gradurstate, 'fvr', 'artvisc')};
% endif

% for i in range(nvars):
% if beta == -0.5:
    fvcomm = ${' + '.join(f'norm_nl[{j}]*fvr[{j}][{i}]' for j in range(ndims))};
% elif beta == 0.5:
    fvcomm = ${' + '.join(f'norm_nl[{j}]*fvl[{j}][{i}]' for j in range(ndims))};
% else:
    fvcomm = ${0.5 + beta}*(${' + '.join(f'norm_nl[{j}]*fvl[{j}][{i}]'
                                         for j in range(ndims))})
           + ${0.5 - beta}*(${' + '.join(f'norm_nl[{j}]*fvr[{j}][{i}]'
                                         for j in range(ndims))});
% endif
% if tau != 0.0:
    fvcomm += ${tau}*(ul[${i}] - ${urstate}[${i}]);
% endif

    ficomm[${i}] += fvcomm;
    ul[${i}] = mag_nl*ficomm[${i}];
% endfor

% if rperiodic:
    // Rotate the common normal flux into the RHS frame
    ${pyfr.expand('rotate', 'rmat', 'ficomm', off=1, transpose=False)};
% endif

% for i in range(nvars):
    ur[${i}] = -mag_nl*ficomm[${i}];
% endfor
</%pyfr:kernel>
