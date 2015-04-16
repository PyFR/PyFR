# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%
    import math
    pi = math.pi
%>

<%pyfr:kernel name='avis' ndim='1'
              s='out view fpdtype_t[${str(nupts)}]'
              amu_e='out view fpdtype_t[${str(nupts)}]'
              amu_f='out view fpdtype_t[${str(nfpts)}]'>
    // Smoothness indicator
    fpdtype_t totEn = 0.0, pnEn = 1e-15, s2;
    fpdtype_t se0= ${math.log10(c['s0'])};

% for i in range(nupts):
    s2 = s[${i}]*s[${i}];
    totEn += s2;

% if ubdegs[i] >= order:
    pnEn += s2;
% endif
% endfor

    fpdtype_t se = log10(pnEn/totEn);

    // Compute cell-wise artificial viscosity
    fpdtype_t mu = (se < se0 - ${c['kappa']})
                 ? 0.0
                 : ${0.5*c['max-amu']}*(1.0 + sin(${0.5*pi/c['kappa']}*(se - se0)));
    mu = (se < se0 + ${c['kappa']}) ? mu : ${c['max-amu']};

    // Copy to all upts/fpts
% for i in range(nupts):
    amu_e[${i}] = mu;
% endfor

% for i in range(nfpts):
    amu_f[${i}] = mu;
% endfor
</%pyfr:kernel>
