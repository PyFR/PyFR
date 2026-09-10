<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

% if s_weights is not None:
## Klockner-Warburton-Hesthaven decay-fit sensor
<%pyfr:kernel name='shocksensor' ndim='1'
              u='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              vtx='out view(${str(nverts)}) reduce(max) fpdtype_t'>
    fpdtype_t tmp;

    // Compute energy per degree level
    fpdtype_t en[${ndeg}] = {};
% for ivdm, deg in zip(invvdm, mode_degs):
    tmp = ${' + '.join(f'{jx}*u[{j}][{svar}]' for j, jx in enumerate(ivdm) if jx != 0)};
    en[${deg}] += tmp*tmp;
% endfor

    // Add baseline decay (KWH eq 4.11) to prevent noise-triggered activation
    fpdtype_t totEn = en[0];
% for d in range(1, ndeg):
    totEn += en[${d}];
% endfor
% for i, d in enumerate(range(1, ndeg)):
    en[${d}] += totEn*${baseline_decay[i]};
% endfor

    // Skyline pessimization (max from high to low degree)
% for d in range(ndeg - 2, -1, -1):
    en[${d}] = fmax(en[${d}], en[${d + 1}]);
% endfor

    // Least-squares decay fit: s = -0.5 * dot(w, ln(en))
    fpdtype_t s = 0;
% for i, d in enumerate(range(1, ndeg)):
    s += ${s_weights[i]}*log(en[${d}]);
% endfor
    s *= ${-0.5};

    // Activation: s < 1 -> max viscosity, s > 3 -> zero
    fpdtype_t mu = (s > ${2.0 + c['kappa']})
                 ? 0.0
                 : ${0.5*c['max-artvisc']}*(1.0 - sin(${0.5*math.pi/c['kappa']}*(s - 2.0)));
    mu = (s < ${2.0 - c['kappa']}) ? ${c['max-artvisc']} : mu;

% for i in range(nverts):
    vtx[${i}] = mu;
% endfor
</%pyfr:kernel>

% else:
## Persson-Peraire modal energy ratio sensor (fallback for low order)
<% se0 = math.log10(c['s0']/order**4) %>

<%pyfr:kernel name='shocksensor' ndim='1'
              u='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              vtx='out view(${str(nverts)}) reduce(max) fpdtype_t'>
    // Smoothness indicator
    fpdtype_t totEn = 0.0, pnEn = 1e-15, tmp;

% for ivdm, bmode in zip(invvdm, ind_modes):
    tmp = ${' + '.join(f'{jx}*u[{j}][{svar}]' for j, jx in enumerate(ivdm) if jx != 0)};

    totEn += tmp*tmp;
% if bmode:
    pnEn += tmp*tmp;
% endif
% endfor

    fpdtype_t se  = ${1/math.log(10)}*log(pnEn/totEn);

    // Compute cell-wise artificial viscosity
    fpdtype_t mu = (se < ${se0 - c['kappa']})
                 ? 0.0
                 : ${0.5*c['max-artvisc']}*(1.0 + sin(${0.5*math.pi/c['kappa']}*(se - ${se0})));
    mu = (se < ${se0 + c['kappa']}) ? mu : ${c['max-artvisc']};

% for i in range(nverts):
    vtx[${i}] = mu;
% endfor
</%pyfr:kernel>
% endif