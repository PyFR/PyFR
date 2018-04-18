# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.acnavstokes.kernels.bcs.common'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t'>
    ur[0] = ul[0];
% for i, v in enumerate(c['v']):
    ur[${i + 1}] = -ul[${i + 1}] + ${2*v};
% endfor
</%pyfr:macro>

<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur' externs='ploc, t'>
    ur[0] = ul[0];
% for i, v in enumerate(c['v']):
    ur[${i + 1}] = ${v};
% endfor
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>
