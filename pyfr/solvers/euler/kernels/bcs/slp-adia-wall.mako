# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur'>

    fpdtype_t nor = 0.0;
% for i in range(ndims):
    nor = nor + ul[${i + 1}]*nl[${i}];
% endfor

    ur[0] = ul[0];
% for i in range(ndims):
    ur[${i + 1}] = ul[${i + 1}] - 2*nor*nl[${i}];
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}];
    
</%pyfr:macro>
