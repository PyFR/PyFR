# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<% eps = 0.001 %>

<%pyfr:macro name='rsolve_t1d' params='ul, ur, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${nvars}], fr[${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}], va[${ndims}], dv[${ndims}];
    fpdtype_t v1,v2,v3;
    fpdtype_t pl, pr, r2a2;

    ${pyfr.expand('inviscid_1dflux', 'ul','fl','pl','vl')};
    ${pyfr.expand('inviscid_1dflux', 'ur','fr','pr','vr')};

    // Compute Roe averaged density and enthalpy
    fpdtype_t roa = sqrt(ul[0])*sqrt(ur[0]);
    fpdtype_t ha = (sqrt(ul[0])*(pr + ur[${ndims + 1}]) + sqrt(ur[0])*(pl + ul[${ndims + 1}]))
                  /(sqrt(ul[0])*ur[0] + sqrt(ur[0])*ul[0]);
    fpdtype_t invsqrulpur = 1 / (sqrt(ul[0]) + sqrt(ur[0]));

% for i in range(ndims):
    va[${i}] = (vl[${i}]*sqrt(ul[0]) + vr[${i}]*sqrt(ur[0]))*invsqrulpur;
% endfor

    fpdtype_t qq = ${pyfr.dot('va[{i}]', 'va[{i}]', i=ndims)};  
    fpdtype_t a = sqrt(${c['gamma'] - 1}*(ha - 0.5*qq));

    // Compute the Eigenvalues
    fpdtype_t l1 = fabs(va[0] - a);
    fpdtype_t l2 = fabs(va[0]);
    fpdtype_t l3 = fabs(va[0] + a);

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
    r2a2 = 1 / (2*a*a);
    v1 = (dp - roa*a*dv[0])*r2a2;
    v2 = dro - dp*2*r2a2;
    v3 = (dp + roa*a*dv[0])*r2a2;
    nf[0] = 0.5*(fl[0] + fr[0]) - (l1*v1 + l2*v2 + l3*v3);
% for i in range(ndims):
% if i == 0:
    v1 = (dp - roa*a*dv[0])*r2a2*(va[${i}] - a);
    v2 = (dro - dp*2*r2a2)*va[${i}];
    v3 = (dp + roa*a*dv[0])*r2a2*(va[${i}] + a);
% else:
    v1 = (dp - roa*a*dv[0])*r2a2*va[${i}];
    v2 = (dro - dp*2*r2a2)*va[${i}] + roa*dv[${i}];
    v3 = (dp + roa*a*dv[0])*r2a2*va[${i}];
% endif
    nf[${i+1}] = 0.5*(fl[${i+1}] + fr[${i+1}]) - (l1*v1 + l2*v2 + l3*v3);
% endfor

    v1 = (dp - roa*a*dv[0])*r2a2*(ha - a*va[0]);
    v2 = (dro - dp*2*r2a2)*qq*0.5 + roa*(${pyfr.dot('va[{i}]', 'dv[{i}]', i=ndims)} - va[0]*dv[0]);
    v3 = (dp + roa*a*dv[0])*r2a2*(ha + a*va[0]);
    nf[${nvars - 1}] = 0.5*(fl[${nvars - 1}] + fr[${nvars - 1}]) - (l1*v1 + l2*v2 + l3*v3);
</%pyfr:macro>

<%include file='pyfr.solvers.euler.kernels.rsolvers.rsolve_trans'/>
