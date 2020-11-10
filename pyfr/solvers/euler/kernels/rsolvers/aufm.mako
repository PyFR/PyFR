# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

// AUFM vector Splitting Approach (SW dU with HLL s2)
<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    fpdtype_t psl[${nvars}], psr[${nvars}], du[${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}];
    fpdtype_t pl, pr;
    
    ${pyfr.expand('inviscid_flux', 'ul', 'fl', 'pl', 'vl')};
    ${pyfr.expand('inviscid_flux', 'ur', 'fr', 'pr', 'vr')};

    // Get the average, left, and right sound speeds
    fpdtype_t cl = sqrt(${c['gamma']}*pl/ul[0]);
    fpdtype_t cr = sqrt(${c['gamma']}*pr/ur[0]);
    fpdtype_t cb = 0.5*(cl + cr);

    // Get the normal left and right velocities
    fpdtype_t nvl = ${pyfr.dot('n[{i}]', 'vl[{i}]', i=ndims)};
    fpdtype_t nvr = ${pyfr.dot('n[{i}]', 'vr[{i}]', i=ndims)};
    
    fpdtype_t ql = ${pyfr.dot('vl[{i}]', 'vl[{i}]', i=ndims)};
    fpdtype_t qr = ${pyfr.dot('vr[{i}]', 'vr[{i}]', i=ndims)};
    
    // Get wave speeds
    fpdtype_t s1 = 0.5*(nvl + nvr);
    fpdtype_t s2 = (s1 > 0.) ? min(0.,max(nvl,nvr)-max(cl,cr)) : max(0.,max(nvl,nvr)+max(cl,cr));

    // Get Mach Number
    fpdtype_t m = s1/(s1 - s2);

    // Get left, right, and delta split vectors 
    psl[0] = 0.;
    psr[0] = 0.;
    du[0] = (pl - pr)/(2.*cb);
% for i in range(ndims):
    psl[${i}+1] = n[${i}]*pl;
    psr[${i}+1] = n[${i}]*pr;
    du[${i}+1] = (pl*vl[${i}] - pr*vr[${i}])/(2.*cb);
% endfor
    psl[${nvars}-1] = pl*nvl; 
    psr[${nvars}-1] = pr*nvr;
    du[${nvars}-1] = ((cb*cb/${c['gamma'] - 1})*(pl-pr) + 0.5*(pl*ql - pr*qr))/(2.0*cb);


    // Output
% for i in range(nvars):
    nf[${i}] = (s1 > 0.) ? (1.-m)*(0.5*(psl[${i}] + psr[${i}]) + du[${i}]) + m*(ul[${i}]*(nvl - s2) + psl[${i}]) :
                           (1.-m)*(0.5*(psl[${i}] + psr[${i}]) + du[${i}]) + m*(ur[${i}]*(nvr - s2) + psr[${i}]);
% endfor
</%pyfr:macro>
