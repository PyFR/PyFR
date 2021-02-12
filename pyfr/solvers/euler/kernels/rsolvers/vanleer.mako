# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.primitives'/>

<% gm = (c['gamma'] - 1) %>
<% gp = (c['gamma'] + 1) %>
<% rg = (1/c['gamma']) %>
<% rgmgp = (1/(gm*gp)) %>

// vn Leer Flux vector Splitting Approach
<%pyfr:macro name='rsolve_t1d' params='ul, ur, nf'>
    fpdtype_t fp[${nvars}], fm[${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}];
    fpdtype_t pl, pr;
    
    ${pyfr.expand('inviscid_prim', 'ul', 'pl', 'vl')};
    ${pyfr.expand('inviscid_prim', 'ur', 'pr', 'vr')};

    // Get left, and right sound speeds and normal Mach
    fpdtype_t cl = sqrt(${c['gamma']}*pl / ul[0]);
    fpdtype_t cr = sqrt(${c['gamma']}*pr / ur[0]);
    fpdtype_t ml = vl[0] / cl;
    fpdtype_t mr = vr[0] / cr;
    
    fpdtype_t hvl = ${pyfr.dot('vl[{i + 1}]', 'vl[{i + 1}]', i=ndims - 1)};
    fpdtype_t hvr = ${pyfr.dot('vr[{i + 1}]', 'vr[{i + 1}]', i=ndims - 1)};

    // Get f+/- mass terms
    fpdtype_t fmp =  0.25*ul[0]*cl*(ml + 1)*(ml + 1);
    fpdtype_t fmm = -0.25*ur[0]*cr*(mr - 1)*(mr - 1);
    
    fp[0] = fmp;
    fm[0] = fmm;

    fp[1] = fmp*(vl[0] + ${rg}*(2*cl - vl[0]));
    fm[1] = fmm*(vr[0] - ${rg}*(2*cr + vr[0]));
% for i in range(1,ndims):
    fp[${i + 1}] = fmp*vl[${i}];
    fm[${i + 1}] = fmm*vr[${i}];
% endfor
    fp[${nvars - 1}] = fmp*(0.5*(${gm}*vl[0] + 2*cl)*(${gm}*vl[0] + 2*cl)*${rgmgp} + 0.5*hvl;
    fm[${nvars - 1}] = fmm*(0.5*(${gm}*vr[0] - 2*cr)*(${gm}*vr[0] - 2*cr)*${rgmgp} + 0.5*hvr;

    // Output
% for i in range(nvars):
    nf[${i}] = (ml > 1) ? fp[${i}] : ((mr < 1) ? fm[${i}] : fm[${i}] + fp[${i}]);
% endfor
</%pyfr:macro>

<%include file='pyfr.solvers.euler.kernels.rsolvers.rsolve_trans'/>
