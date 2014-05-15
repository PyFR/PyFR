# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur'>

<% gamma, v = c['gamma'], c['v'] %>
<% cs, s = (c['gamma']*c['p']/c['rho'])**0.5, c['p']/(c['rho']**c['gamma']) %>
<% gmo, ratio = c['gamma'] - 1.0, 2.0*(c['gamma']*c['p']/c['rho'])**0.5/(c['gamma'] - 1.0) %>
                        
    fpdtype_t V_i = 0.0;
% for i in range(ndims):
    V_i = V_i + ul[${i + 1}]*nl[${i}];
% endfor
    V_i = V_i/ul[0];

    fpdtype_t V_e = 0.0;
% for i in range(ndims):
    V_e = V_e + ${v[i]}*nl[${i}];
% endfor
    
    fpdtype_t p_i = (ul[${nvars - 1}]
                  - 0.5*(1.0/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))})*${gmo};
    fpdtype_t c_i = sqrt(${gamma}*p_i/ul[0]);
    fpdtype_t R_e = (fabs(V_e) >= ${cs} && V_i >= 0) ? V_i - 2.0*c_i/${gmo} : V_e - ${ratio};
    fpdtype_t R_i = (fabs(V_e) >= ${cs} && V_i < 0) ? V_e + ${ratio} : V_i + 2.0*c_i/${gmo};
    fpdtype_t V_b = (R_e + R_i)/2.0;
    fpdtype_t c_b = ${gmo}*(R_i - R_e)/4.0;
    fpdtype_t s_i = pow(ul[0], ${-gmo})*c_i*c_i/${gamma};
    fpdtype_t rho_b = (V_i < 0) ? pow(c_b*c_b/${gamma*s}, ${1.0/gmo}) :
                    pow(c_b*c_b/(${gamma}*s_i), ${1.0/gmo});
    fpdtype_t p_b = rho_b*c_b*c_b/${gamma};

    ur[0] = rho_b;
% for i in range(ndims):
    ur[${i + 1}] = (V_i >= 0) ? rho_b*(ul[${i + 1}]/ul[0] + (V_b - V_i)*nl[${i}]) :
                 rho_b*(${v[i]} + (V_b - V_e)*nl[${i}]); 
% endfor
    ur[${nvars - 1}] = p_b/${gmo} + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
                     
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_zero'/>