# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% se0 = math.log10(c['s0']) %>

<%pyfr:kernel name='shocksensor' ndim='1'
              u='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              artvisc='out fpdtype_t'>
    // Smoothness indicator
    fpdtype_t totEn = 0.0, pnEn = 1e-15, tmp;

% for i, deg in enumerate(ubdegs):
    tmp = ${' + '.join('{jx}*u[{j}][{svar}]'.format(j=j, jx=jx, svar=svar)
                       for j, jx in enumerate(invvdm[i]) if jx != 0)};

    totEn += tmp*tmp;
% if deg >= order:
    pnEn += tmp*tmp;
% endif
% endfor

    fpdtype_t se  = ${1/math.log(10)}*log(pnEn/totEn);

    // Compute cell-wise artificial viscosity
    fpdtype_t mu = (se < ${se0 - c['kappa']})
                 ? 0.0
                 : ${0.5*c['max-artvisc']}*(1.0 + sin(${0.5*math.pi/c['kappa']}*(se - ${se0})));
    mu = (se < ${se0 + c['kappa']}) ? mu : ${c['max-artvisc']};

    artvisc = mu;
</%pyfr:kernel>
