# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<% eps = 0.001 %>

<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}], va[${ndims}], dv[${ndims}];
    fpdtype_t v1[${nvars}], v2[${nvars}], v3[${nvars}];
    fpdtype_t pl, pr, r2a2;

    ${pyfr.expand('inviscid_flux', 'ul', 'fl', 'pl', 'vl')};
    ${pyfr.expand('inviscid_flux', 'ur', 'fr', 'pr', 'vr')};

    // Get the normal velocity jump
    fpdtype_t dvs = ${pyfr.dot('n[{i}]', 'vr[{i}] - vl[{i}]', i=ndims)};

    // Compute Roe averaged density and enthalpy
    fpdtype_t roa = sqrt(ul[0])*sqrt(ur[0]);
    fpdtype_t ha = (sqrt(ul[0])*(pr + ur[${ndims + 1}])
                 + sqrt(ur[0])*(pl + ul[${ndims + 1}]))
                / (sqrt(ul[0])*ur[0] + sqrt(ur[0])*ul[0]);
    fpdtype_t invsqrulpur = 1/(sqrt(ul[0]) + sqrt(ur[0]));

% for i in range(ndims):
    va[${i}] = (vl[${i}]*sqrt(ul[0]) + vr[${i}]*sqrt(ur[0]))*invsqrulpur;
% endfor

    fpdtype_t qq = ${pyfr.dot('va[{i}]', 'va[{i}]', i=ndims)};
    fpdtype_t vs = ${pyfr.dot('n[{i}]', 'va[{i}]', i=ndims)};
    fpdtype_t a = sqrt(${c['gamma'] - 1}*(ha - 0.5*qq));

    // Compute the Eigenvalues
    fpdtype_t l1 = fabs(vs - a);
    fpdtype_t l2 = fabs(vs);
    fpdtype_t l3 = fabs(vs + a);

    // Entropy fix
    l1 = (l1 < ${eps}) ? ${1/(2*eps)}*(l1*l1 + ${eps**2}) : l1;
    l3 = (l3 < ${eps}) ? ${1/(2*eps)}*(l3*l3 + ${eps**2}) : l3;

    // Get the jumps 
% for i in range(ndims):
    dv[${i}] = vr[${i}] - vl[${i}];
% endfor

    fpdtype_t dro = ur[0] - ul[0];
    fpdtype_t dp = pr - pl;

    // Compute the Eigenvectors
    r2a2 = 1/(2*a*a);
    v1[0] = (dp - roa*a*dvs)*r2a2;
    v1[${nvars - 1}] = (dp - roa*a*dvs)*r2a2*(ha - a*vs);
    v2[0] = dro - dp*2*r2a2;
    v2[${nvars - 1}] = (dro - dp*2*r2a2)*qq*0.5 + roa*(${pyfr.dot('va[{i}]', 'dv[{i}]', i=ndims)} - vs*dvs);
    v3[0] = (dp + roa*a*dvs)*r2a2;
    v3[${nvars - 1}] = (dp + roa*a*dvs)*r2a2*(ha + a*vs);

% for i in range(ndims):
    v1[${i + 1}] = (dp - roa*a*dvs)*r2a2*(va[${i}] - a*n[${i}]);
    v2[${i + 1}] = (dro - dp*2*r2a2)*va[${i}] + roa*(dv[${i}] - dvs*n[${i}]);
    v3[${i + 1}] = (dp + roa*a*dvs)*r2a2*(va[${i}] + a*n[${i}]);
% endfor

    // Output
% for i in range(nvars):
    nf[${i}] = 0.5*(${' + '.join('n[{j}]*(fl[{j}][{i}] + fr[{j}][{i}])'
                                 .format(i=i, j=j) for j in range(ndims))}
             - (l1*v1[${i}] + l2*v2[${i}] + l3*v3[${i}]));
% endfor
</%pyfr:macro>
