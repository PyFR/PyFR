# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

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
    ur[${nvars - 1}] = ${c['p']/(c['gamma'] - 1) +\
                       0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>
