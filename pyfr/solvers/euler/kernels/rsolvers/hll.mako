# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}];
    fpdtype_t pl, pr;
    fpdtype_t nf_sub, nf_fl, nf_fr;

    ${pyfr.expand('inviscid_flux', 'ul', 'fl', 'pl', 'vl')};
    ${pyfr.expand('inviscid_flux', 'ur', 'fr', 'pr', 'vr')};

    // Roe average velocity and take normal
    fpdtype_t nv = ${pyfr.dot('n[{i}]',
                     'sqrt(ul[0])*vl[{i}] + sqrt(ur[0])*vr[{i}]', i=ndims)}
                     / (sqrt(ul[0]) + sqrt(ur[0]));

    // Roe average enthalpy
    fpdtype_t H = (ur[0]*sqrt(ul[0])*(ul[${ndims + 1}] + pl)
                   + (ul[0]*sqrt(ur[0])*ur[${ndims + 1}] + pr))
                   / (ul[0]*ur[0]*(sqrt(ul[0]) + sqrt(ur[0])));

    // Roe average sound speed
    fpdtype_t a = sqrt(${c['gamma']-1}*(H - 0.5*nv*nv));

    // Estimate the left and right wave speed, sl and sr
    fpdtype_t sl = nv - a;
    fpdtype_t sr = nv + a;
    fpdtype_t rcpsrsl = 1/(sr - sl);

    // Output
    // Possible solutions for flux
% for i in range(nvars):
    nf_fl = (${' + '.join('n[{j}]*fl[{j}][{i}]'
                                 .format(i=i, j=j) for j in range(ndims))});
    nf_fr = (${' + '.join('n[{j}]*fr[{j}][{i}]'
                                 .format(i=i, j=j) for j in range(ndims))});
    nf_sub = (sr*nf_fl - sl*nf_fr + sl*sr*(ur[${i}] - ul[${i}]))*rcpsrsl;
    nf[${i}] = (0 <= sl) ? nf_fl : (0 >= sr) ? nf_fr : nf_sub;
% endfor
</%pyfr:macro>
