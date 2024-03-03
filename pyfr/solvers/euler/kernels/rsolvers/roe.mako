<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<% eps = 0.001 %>

<%pyfr:macro name='rsolve_1d' params='ul, ur, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${nvars}], fr[${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}];
    fpdtype_t pl, pr;
    fpdtype_t va[${ndims}], dv[${ndims}];

    ${pyfr.expand('inviscid_flux_1d', 'ul', 'fl', 'pl', 'vl')};
    ${pyfr.expand('inviscid_flux_1d', 'ur', 'fr', 'pr', 'vr')};

    // Compute Roe averaged density and enthalpy
    fpdtype_t roa = sqrt(ul[0])*sqrt(ur[0]);
    fpdtype_t ha = (sqrt(ul[0])*(pr + ur[${nvars - 1}]) +
                    sqrt(ur[0])*(pl + ul[${nvars - 1}])) /
                    (sqrt(ul[0])*ur[0] + sqrt(ur[0])*ul[0]);

    fpdtype_t inv_rar = 1 / (sqrt(ul[0]) + sqrt(ur[0]));
% for i in range(ndims):
    va[${i}] = (vl[${i}]*sqrt(ul[0]) + vr[${i}]*sqrt(ur[0])) * inv_rar;
% endfor

    fpdtype_t qq = ${pyfr.dot('va[{i}]', i=ndims)};
    fpdtype_t a = sqrt(${c['gamma'] - 1}*(ha - 0.5*qq));

    // Compute the Eigenvalues
    fpdtype_t l1 = fabs(va[0] - a);
    fpdtype_t l2 = fabs(va[0]);
    fpdtype_t l3 = fabs(va[0] + a);

    // Entropy fix
    l1 = (l1 < ${eps}) ? ${0.5 / eps}*(l1*l1 + ${eps**2}) : l1;
    l3 = (l3 < ${eps}) ? ${0.5 / eps}*(l3*l3 + ${eps**2}) : l3;

    // Calculate the jump values
% for i in range(ndims):
    dv[${i}] = vr[${i}] - vl[${i}];
% endfor
    fpdtype_t dro = ur[0] - ul[0];
    fpdtype_t dp = pr - pl;

    fpdtype_t inv_a2 = 1 / (2*a*a);

    // Compute the mass eigenvectors
    fpdtype_t v1 = (dp - roa*a*dv[0])*inv_a2;
    fpdtype_t v2 = dro - dp*2*inv_a2;
    fpdtype_t v3 = (dp + roa*a*dv[0])*inv_a2;
    nf[0] = 0.5*(fl[0] + fr[0]) - (l1*v1 + l2*v2 + l3*v3);

    // Compute the momentum eigenvectors
% for i in range(ndims):
% if i == 0:
    v1 = (dp - roa*a*dv[0])*inv_a2*(va[${i}] - a);
    v2 = (dro - dp*2*inv_a2)*va[${i}];
    v3 = (dp + roa*a*dv[0])*inv_a2*(va[${i}] + a);
% else:
    v1 = (dp - roa*a*dv[0])*inv_a2*va[${i}];
    v2 = (dro - dp*2*inv_a2)*va[${i}] + roa*dv[${i}];
    v3 = (dp + roa*a*dv[0])*inv_a2*va[${i}];
% endif
    nf[${i + 1}] = 0.5*(fl[${i + 1}] + fr[${i + 1}]) - (l1*v1 + l2*v2 + l3*v3);
% endfor

    // Compute the energy eigenvectors
    v1 = (dp - roa*a*dv[0])*inv_a2*(ha - a*va[0]);
    v2 = roa*(${pyfr.dot('va[{i}]', 'dv[{i}]', i=ndims)} - va[0]*dv[0]) +
         (dro - dp*2*inv_a2)*qq*0.5;
    v3 = (dp + roa*a*dv[0])*inv_a2*(ha + a*va[0]);
    nf[${nvars - 1}] = 0.5*(fl[${nvars - 1}] + fr[${nvars - 1}]) - (l1*v1 + l2*v2 + l3*v3);
</%pyfr:macro>

<%include file='pyfr.solvers.euler.kernels.rsolvers.rsolve1d'/>
