# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='inviscid_flux' params='s, f'>
    
% for i in range(ndims):
    f[${i}][0] = ${c['a'][i]}*s[0];
% endfor

</%pyfr:macro>
