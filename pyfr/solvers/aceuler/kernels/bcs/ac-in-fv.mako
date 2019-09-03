# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t'>
    ur[0] = ul[0];

% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = ${c[v]};
% endfor
</%pyfr:macro>
