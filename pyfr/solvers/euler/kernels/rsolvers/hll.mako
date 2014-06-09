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

    // Get the normal left and right velocities
    fpdtype_t nvl = ${pyfr.dot('n[{i}]', 'vl[{i}]', i=ndims)};
    fpdtype_t nvr = ${pyfr.dot('n[{i}]', 'vr[{i}]', i=ndims)};

    // Compute the Roe-averaged velocity
    fpdtype_t nv = (sqrt(ul[0])*nvl + sqrt(ur[0])*nvr)
                 / (sqrt(ul[0]) + sqrt(ur[0]));

    // Compute the Roe-averaged enthalpy
    fpdtype_t H = (sqrt(ul[0])*(pr + ur[${ndims + 1}])
                 + sqrt(ur[0])*(pl + ul[${ndims + 1}]))
                / (sqrt(ul[0])*ur[0] + sqrt(ur[0])*ul[0]);

    // Roe average sound speed
    fpdtype_t a = sqrt(${c['gamma'] - 1}*(H - 0.5*nv*nv));

    // Estimate the left and right wave speed, sl and sr
    fpdtype_t sl = nv - a;
    fpdtype_t sr = nv + a;
    fpdtype_t rcpsrsl = 1/(sr - sl);

    // Output
% for i in range(nvars):
    nf_fl = ${' + '.join('n[{j}]*fl[{j}][{i}]'.format(i=i, j=j)
                         for j in range(ndims))};
    nf_fr = ${' + '.join('n[{j}]*fr[{j}][{i}]'.format(i=i, j=j)
                         for j in range(ndims))};
    nf_sub = (sr*nf_fl - sl*nf_fr + sl*sr*(ur[${i}] - ul[${i}]))*rcpsrsl;
    nf[${i}] = (0 <= sl) ? nf_fl : (0 >= sr) ? nf_fr : nf_sub;
% endfor
</%pyfr:macro>
