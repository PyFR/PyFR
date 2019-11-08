# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t'>
    ur[0] = ${c['rho']};
% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = (${c['rho']})*(${c[v]});
% endfor
    ur[${nvars - 1}] = ${c['p']}/${c['gamma'] - 1} +
                       0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>
