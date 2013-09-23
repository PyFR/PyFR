# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:function name='bc_rsolve_state'
                params='const fpdtype_t ul[${str(nvars)}],
                        fpdtype_t ur[${str(nvars)}]'>
    ur[0] = ${c['rho']};
% for i in range(ndims):
    ur[${i + 1}] = ${c['rho']*c['v'][i]};
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}]
                     - 0.5*(1.0/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))}
                     + ${0.5*c['rho']*sum(v**2 for v in c['v'])};
</%pyfr:function>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_zero'/>
