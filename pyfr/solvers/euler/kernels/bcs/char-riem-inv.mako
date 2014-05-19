# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% gmo = c['gamma'] - 1.0 %>
<% gamma, v = c['gamma'], c['v'] %>
<% cs, s = (c['gamma']*c['p']/c['rho'])**0.5, c['p']/(c['rho']**c['gamma']) %>
<% ratio = 2.0*(c['gamma']*c['p']/c['rho'])**0.5/(c['gamma'] - 1.0) %>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur'>
    fpdtype_t inv = 1.0/ul[0];
    fpdtype_t V_e = ${' + '.join('{0}*nl[{1}]'.format(v[i], i)
                                 for i in range(ndims))};
    fpdtype_t V_i = inv*(${' + '.join('ul[{1}]*nl[{0}]'.format(i, i + 1)
                                      for i in range(ndims))});
    fpdtype_t p_i = (ul[${nvars - 1}]
                    - 0.5*inv*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))})*${gmo};
    fpdtype_t c_i = sqrt(${gamma}*p_i*inv);
    fpdtype_t R_e = (fabs(V_e) >= ${cs} && V_i >= 0) ? V_i - c_i*${2.0/gmo} :
                    V_e - ${ratio};
    fpdtype_t R_i = (fabs(V_e) >= ${cs} && V_i < 0) ? V_e + ${ratio} :
                     V_i + c_i*${2.0/gmo};
    fpdtype_t V_b = 0.5*(R_e + R_i);
    fpdtype_t c_b = ${0.25*gmo}*(R_i - R_e);
    fpdtype_t s_i = pow(ul[0], ${-gmo})*c_i*c_i*${1.0/gamma};
    fpdtype_t rho_b = pow((V_i < 0) ? c_b*c_b*${1.0/(gamma*s)} :
                      c_b*c_b*${1.0/gamma}/s_i, ${1.0/gmo});
    fpdtype_t p_b = rho_b*c_b*c_b*${1.0/gamma};

    ur[0] = rho_b;
% for i in range(ndims):
    ur[${i + 1}] = (V_i >= 0) ? rho_b*(ul[${i + 1}]*inv + (V_b - V_i)*nl[${i}]) :
                 rho_b*(${v[i]} + (V_b - V_e)*nl[${i}]); 
% endfor
    ur[${nvars - 1}] = p_b*${1.0/gmo} +
                     0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};                     
</%pyfr:macro>
