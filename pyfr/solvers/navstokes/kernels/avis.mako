# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%
    import math
    pi = math.pi
%>


<%pyfr:kernel name='avis' ndim='1'
              s='in fpdtype_t[${str(nupts)}]'
              amu_e='out fpdtype_t[${str(nrow_amu)}]'
              amu_f='out fpdtype_t[${str(nfpts)}]'>
    // Smoothness indicator
    fpdtype_t totEn = 0.0, pnEn = 1e-15;
    fpdtype_t se0 = ${math.log10(c['s0'])};

% for i, deg in enumerate(ubdegs):
    totEn += s[${i}]*s[${i}];
% if deg >= order:
    pnEn += s[${i}]*s[${i}];
% endif
% endfor

    fpdtype_t se = log10(pnEn/totEn);

    // Compute cell-wise artificial viscosity
    fpdtype_t mu = (se < se0 - ${c['kappa']})
                 ? 0.0
                 : ${0.5*c['max-amu']}*(1.0 + sin(${0.5*pi/c['kappa']}*(se - se0)));
    mu = (se < se0 + ${c['kappa']}) ? mu : ${c['max-amu']};

    // Copy to all upts (or qpts) and fpts
% for i in range(nrow_amu):
    amu_e[${i}] = mu;
% endfor

% for i in range(nfpts):
    amu_f[${i}] = mu;
% endfor
</%pyfr:kernel>
