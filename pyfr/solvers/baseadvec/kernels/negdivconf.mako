<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
% for mod, name in srcmacros:
    <%include file='${mod}'/>
% endfor

<%pyfr:kernel name='negdivconf' ndim='2'
              t='scalar fpdtype_t'
              tdivtconf='inout fpdtype_t[${str(nvars)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              u='in fpdtype_t[${str(nvars)}]'
              rcpdjac='in fpdtype_t'>
fpdtype_t src[${nvars}] = {};

% for mod, name in srcmacros:
    ${pyfr.expand(name, 't', 'u', 'ploc', 'src')};
% endfor

% for i in range(nvars):
    tdivtconf[${i}] = -rcpdjac*tdivtconf[${i}] + src[${i}];
% endfor
</%pyfr:kernel>
