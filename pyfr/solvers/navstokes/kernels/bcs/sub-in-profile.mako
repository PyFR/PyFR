<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>
<% gamma = c['gamma'] %>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='profile'>
    // profile 外部数据: [rho, u, v, w, p] —— 已插值到当前面通量点
    fpdtype_t rho_b = profile[0];
    fpdtype_t p_b = profile[${ndims + 1}];

    ur[0] = rho_b;
% for i in range(ndims):
    ur[${i + 1}] = rho_b*profile[${i + 1}];
% endfor
    ur[${nvars - 1}] = p_b*${1.0/(gamma - 1)}
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_zero'/>
