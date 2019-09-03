# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t'>
    fpdtype_t zeta = ${c['bc-ac-zeta']};

    fpdtype_t V_e = ${' + '.join('{0}*nl[{1}]'.format(c['uvw'[i]], i)
                                 for i in range(ndims))};
    fpdtype_t V_i = ${' + '.join('ul[{0}]*nl[{1}]'.format(i + 1, i)
                                 for i in range(ndims))};

    fpdtype_t C_i = sqrt(V_i*V_i + zeta);
    fpdtype_t C_e = sqrt(V_e*V_e + zeta);

    fpdtype_t R_i = ul[0] + 0.5*(V_i*(V_i + C_i) + zeta*log(V_i + C_i));
    fpdtype_t R_e = ${c['p']} + 0.5*(V_e*(V_e - C_e) - zeta*log(V_e + C_e));

    fpdtype_t V_b = V_e;
    fpdtype_t c, f, df;

% for i in range(c['niters']):
    c = sqrt(V_b*V_b + zeta);
    f = c*V_b + zeta*log(V_b + c) + R_e - R_i;
    df = 2.0*(V_b*V_b + zeta)/c;
    V_b = V_b - f/df;
% endfor

% for i in range(ndims):
    ur[${i + 1}] = (V_i >= 0)
                 ? (ul[${i + 1}]+ (V_b - V_i)*nl[${i}])
                 : (${c['uvw'[i]]} + (V_b - V_e)*nl[${i}]);
% endfor

    fpdtype_t C_b = sqrt(V_b*V_b + zeta);
    ur[0] = R_e - 0.5*(V_b*(V_b - C_b) - zeta*log(V_b + C_b));
</%pyfr:macro>
