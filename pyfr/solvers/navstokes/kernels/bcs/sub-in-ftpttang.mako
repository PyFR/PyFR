# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t'>
    fpdtype_t pl = ${c['gamma'] - 1.0}*(ul[${nvars - 1}]
                 - (0.5/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))});
    fpdtype_t udotu = ${2.0*c['cpTt']}*(1.0
                    - ${c['pt']**(-c['Rdcp'])}*pow(pl, ${c['Rdcp']}));

    ur[0] = ${1.0/c['Rdcp']}*pl/(${c['cpTt']} - 0.5*udotu);
% for i, v in enumerate(c['vc']):
    ur[${i + 1}] = ${v}*ur[0]*sqrt(udotu);
% endfor
    ur[${nvars - 1}] = ${1.0/(c['gamma'] - 1.0)}*pl + 0.5*ur[0]*udotu;
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>
