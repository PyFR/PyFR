# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.aceuler.kernels.flux1d'/>

<% eps = 0.001 %>
<% zeta = c['ac-zeta'] %>
<% rzeta = 1./zeta %>

<%pyfr:macro name='rsolve_t1d' params='ul, ur, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${nvars}], fr[${nvars}];
    fpdtype_t va[${ndims}], dv[${ndims}];
    fpdtype_t v1[${nvars}], v2[${nvars}], v3[${nvars}];
    
    ${pyfr.expand('inviscid_1dflux', 'ul','fl')};
    ${pyfr.expand('inviscid_1dflux', 'ur','fr')};

    // Average velocities
% for i in range(ndims):
    va[${i}] = 0.5*(ul[${i+1}] + ur[${i+1}]);
% endfor

    // Quanitiy jumps
    fpdtype_t dpz = ${rzeta}*(ur[0] - ul[0]); 
% for i in range(ndims):
    dv[${i}] = ur[${i+1}] - ul[${i+1}];
% endfor

    // ACM speed of sound
    fpdtype_t aa = sqrt(va[0]*va[0] + ${zeta});
    fpdtype_t ra = 1./aa;
    
    // Eigenvalues
    fpdtype_t l1 = fabs(va[0] - aa);
    fpdtype_t l2 = fabs(va[0]);
    fpdtype_t l3 = fabs(va[0] + aa);

    // Entropy fix
    //l1 = (l1 < ${eps}) ? ${1/(2*eps)}*(l1*l1 + ${eps**2}) : l1;
    //l3 = (l3 < ${eps}) ? ${1/(2*eps)}*(l3*l3 + ${eps**2}) : l3;

    // Alpha terms
    fpdtype_t a1 = 0.5*ra*(dpz*(va[0] + aa) - dv[0]);
    fpdtype_t a3 = 0.5*ra*(dv[0] - dpz*(va[0] - aa));

    // Compute the Eigenvectors
    fpdtype_t v2c = (2.*a1*dv[0] - dpz*(va[0] + aa))*ra;
    
    v1[0] = a1*${zeta};
    v2[0] = 0.;
    v3[0] = a3*${zeta};
% for i in range(ndims):
% if i == 0:
    v1[${i + 1}] = a1*(aa - va[0]);
    v2[${i + 1}] = 0.;
    v3[${i + 1}] = a3*(aa + va[0]);
% else:
    v1[${i + 1}] = a1*ra*(aa - va[0])*va[${i}];
    v2[${i + 1}] = dv[${i}] + va[${i}]*v2c;
    v3[${i + 1}] = a3*ra*(va[0] + aa)*va[${i}];
% endif
% endfor

    // Output
% for i in range(nvars):
    nf[${i}] = 0.5*(fl[${i}] + fr[${i}]) - 0.5*(l1*v1[${i}] + l2*v2[${i}] + l3*v3[${i}]);
% endfor

</%pyfr:macro>

<%include file='pyfr.solvers.aceuler.kernels.rsolvers.rsolve_trans'/>
