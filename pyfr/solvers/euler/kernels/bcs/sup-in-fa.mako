# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, ur'>
    ur[0] = ${c['rho']};
% for i in range(ndims):
    ur[${i + 1}] = ${c['rho']*c['v'][i]};
% endfor
    ur[${nvars - 1}] = ${c['p']/(c['gamma'] - 1) +\
                         0.5*c['rho']*sum(v**2 for v in c['v'])};
</%pyfr:macro>
