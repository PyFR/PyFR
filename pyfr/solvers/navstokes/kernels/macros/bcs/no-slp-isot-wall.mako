<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.macros.bcs.common'/>
<%include file='pyfr.solvers.euler.kernels.macros.bcs.slp-adia-wall'/>

<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur' externs='ploc, t'>
    ur[0] = ul[0];
% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = ${c[v]}*ul[0];
% endfor
    ur[${nvars - 1}] = ${c['cpTw']/c['gamma']}*ur[0]
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_wall'/>
