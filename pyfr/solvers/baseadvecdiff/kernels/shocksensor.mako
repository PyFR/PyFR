# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='shocksensor' ndim='1'
              s='in fpdtype_t[${str(nupts)}]'
              amu='out fpdtype_t'>
    // Smoothness indicator
    fpdtype_t totEn = 0.0, pnEn = 1e-15;

% for i, deg in enumerate(ubdegs):
    totEn += s[${i}]*s[${i}];
% if deg >= order:
    pnEn += s[${i}]*s[${i}];
% endif
% endfor

    fpdtype_t se0 = ${math.log10(c['s0'])};
    fpdtype_t se  = log10(pnEn/totEn);

    // Compute cell-wise artificial viscosity
    fpdtype_t mu = (se < se0 - ${c['kappa']})
                 ? 0.0
                 : ${0.5*c['max-amu']}*(1.0 + sin(${0.5*math.pi/c['kappa']}*(se - se0)));
    mu = (se < se0 + ${c['kappa']}) ? mu : ${c['max-amu']};

    amu = mu;
</%pyfr:kernel>
