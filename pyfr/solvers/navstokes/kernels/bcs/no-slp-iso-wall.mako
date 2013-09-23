# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:function name='bc_rsolve_state'
                params='const fpdtype_t ul[${str(nvars)}],
                        fpdtype_t ur[${str(nvars)}]'>
    ur[0] = ul[0];
% for i in range(ndims):
    ur[${i + 1}] = -ul[${i + 1}];
% endfor
    ur[${nvars - 1}] = ${c['cpTw']/c['gamma']}*ul[0]
                     + 0.5*(1.0/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))};
</%pyfr:function>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>
