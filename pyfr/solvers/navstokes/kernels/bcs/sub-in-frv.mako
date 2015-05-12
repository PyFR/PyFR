# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur, ploc, t'>
% for i in range(ndims):
    fpdtype_t ${'xyz'[i]} = ploc[${i}];
% endfor
    fpdtype_t rho = (${c['rho']});

% for i in range(ndims):
    fpdtype_t ${'uvw'[i]} = (${c['uvw'[i]]});
% endfor

    ur[0] = rho;
% for i in range(ndims):
    ur[${i + 1}] = rho * ${'uvw'[i]};
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}]
                     - 0.5*(1.0/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))}
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_zero'/>
