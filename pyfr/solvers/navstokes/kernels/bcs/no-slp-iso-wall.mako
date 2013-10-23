# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:function name='bc_rsolve_state'
                params='const fpdtype_t ul[${str(nvars)}],
                        fpdtype_t ur[${str(nvars)}]'>
    ur[0] = ul[0];
% for i, v in enumerate(c['v']):
    ur[${i + 1}] = -ul[${i + 1}] + 2*ul[0]*${v};
% endfor
    ur[${nvars - 1}] = ${c['cpTw']/c['gamma']}*ul[0]
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:function>

<%pyfr:function name='bc_ldg_state'
                params='const fpdtype_t ul[${str(nvars)}],
                        fpdtype_t ur[${str(nvars)}]'>
    ur[0] = ul[0];
% for i, v in enumerate(c['v']):
    ur[${i + 1}] = ul[0]*${v};
% endfor
    ur[${nvars - 1}] = ${c['cpTw']/c['gamma']}*ul[0]
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:function>

<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>
