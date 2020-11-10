# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.aceuler.kernels.flux'/>

<% zeta = c['ac-zeta'] %>
<% rzeta = 1./zeta %>

<%pyfr:macro name='rsolve_t1d' params='ul, ur, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${nvars}], fr[${nvars}];
    fpdtype_t v1, v2, v3;
    
    ${pyfr.expand('inviscid_1dflux', 'ul','fl')};
    ${pyfr.expand('inviscid_1dflux', 'ur','fr')};

    // Average velocity in x
    fpdtype_t vax = 0.5*(ul[1] + ur[1]);

    // Quanitiy jumps
    fpdtype_t dpz = ${rzeta}*(ur[0] - ul[0]); 
    fpdtype_t dvx = ur[1] - ul[1];

    // ACM speed of sound
    fpdtype_t aa = sqrt(vax*vax + ${zeta});
    fpdtype_t ra = 1./aa;
    
    // Eigenvalues
    fpdtype_t l1 = fabs(vax - aa);
    fpdtype_t l2 = fabs(vax);
    fpdtype_t l3 = fabs(vax + aa);

    // Alpha terms
    fpdtype_t a1 =  0.5*ra*(dpz*(vax + aa) - dvx);
    fpdtype_t a3 = -0.5*ra*(dpz*(vax - aa) - dvx);

    // Compute the Eigenvectors
    fpdtype_t vc = (a1*(vax - aa) - a3*(vax + aa))*ra;
    
    v1 = a1*${zeta};
    v2 = 0.;
    v3 = a3*${zeta};
    nf[0] = 0.5*(fl[0] + fr[0]) - 0.5*(l1*v1 + l2*v2 + l3*v3);
% for i in range(ndims):
% if i == 0:
    v1 = a1*(vax - aa);
    v2 = 0.;
    v3 = a3*(vax + aa);
% else:
    v1 = -a1*ra*(vax - aa)*0.5*(ul[${i + 1}] + ur[${i + 1}]);
    v2 =  (ur[${i + 1}] - ul[${i + 1}]) + vax*vc;
    v3 =  a3*ra*(vax + aa)*0.5*(ul[${i + 1}] + ur[${i + 1}]);
% endif
    nf[${i + 1}] = 0.5*(fl[${i + 1}] + fr[${i + 1}]) - 0.5*(l1*v1 + l2*v2 + l3*v3);
% endfor

</%pyfr:macro>

<%include file='pyfr.solvers.aceuler.kernels.rsolvers.rsolve_trans'/>
