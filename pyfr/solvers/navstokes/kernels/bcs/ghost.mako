<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<% tau = c['ldg-tau'] %>

<% bc_ldg_state = ('bc_ldg_state', 'ul', 'nl', 'ur') %>
<% bc_rsolve_state = ('bc_rsolve_state', 'ul', 'nl', 'ur') %>
<% params = 'ul, gradul, amul, nl, magnl' %>

% if stv:
<% params += ", ploc, t" %>
<% bc_ldg_state = bc_ldg_state + ('ploc', 't') %>
<% bc_rsolve_state = bc_rsolve_state + ('ploc', 't') %>
% endif

<%pyfr:macro name='bc_common_flux_state' params="${params}" >

    // Viscous states
    fpdtype_t ur[${nvars}], gradur[${ndims}][${nvars}];
    ${pyfr.expand(*bc_ldg_state)};
    ${pyfr.expand('bc_ldg_grad_state', 'ul', 'nl', 'gradul', 'gradur')};

    fpdtype_t fvr[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ur', 'gradur', 'amul', 'fvr')};

    // Inviscid (Riemann solve) state
    ${pyfr.expand(*bc_rsolve_state)};

    // Perform the Riemann solve
    fpdtype_t ficomm[${nvars}], fvcomm;
    ${pyfr.expand('rsolve', 'ul', 'ur', 'nl', 'ficomm')};

% for i in range(nvars):
    fvcomm = ${' + '.join('nl[{j}]*fvr[{j}][{i}]'.format(i=i, j=j)
                          for j in range(ndims))};
% if tau != 0.0:
    fvcomm += ${tau}*(ul[${i}] - ur[${i}]);
% endif

    ul[${i}] = magnl*(ficomm[${i}] + fvcomm);
% endfor
</%pyfr:macro>
