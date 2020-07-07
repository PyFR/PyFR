# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux1d'/>

<%pyfr:macro name='rsolve_t1d' params='ul, ur, nf'>
    fpdtype_t fl[${nvars}], fr[${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}];
    fpdtype_t pl, pr;
    fpdtype_t nf_sub;

    // Compute the left and right fluxes + velocities and pressures
    ${pyfr.expand('inviscid_1dflux', 'ul', 'fl', 'pl', 'vl')};
    ${pyfr.expand('inviscid_1dflux', 'ur', 'fr', 'pr', 'vr')};

    // Compute the mean velocity
    fpdtype_t nv = 0.5*(vl[0] + vr[0]);

    // Einfeldt modified sound speed
    fpdtype_t ravrho = 1./(sqrt(ul[0]) + sqrt(ur[0]));
    fpdtype_t eta = 0.5*sqrt(ul[0]*ur[0])*ravrho*ravrho;
    fpdtype_t d = (${c['gamma']}*pl/sqrt(ul[0]) + ${c['gamma']}*pr/sqrt(ur[0]))*ravrho +
    	      eta*(vr[0] - vl[0])*(vr[0] - vl[0]);
    d = sqrt(d);

    // Estimate the left and right wave speed, sl and sr
    fpdtype_t sl = nv - d;
    fpdtype_t sr = nv + d;
    fpdtype_t rcpsrsl = 1/(sr - sl);

    // Output
% for i in range(nvars):
    nf_sub = (sr*fl[${i}] - sl*fr[${i}] + sl*sr*(ur[${i}] - ul[${i}]))*rcpsrsl;
    
    nf[${i}] = (0 <= sl) ? fl[${i}] : (0 >= sr) ? fr[${i}] : nf_sub;
% endfor
</%pyfr:macro>

<%include file='pyfr.solvers.euler.kernels.rsolvers.rsolve_trans'/>
