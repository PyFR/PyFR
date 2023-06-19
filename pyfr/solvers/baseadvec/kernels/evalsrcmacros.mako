<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
% for mod, name in srcmacros:
    <%include file='${mod}'/>
% endfor

<%pyfr:kernel name='evalsrcmacros' ndim='2'
              t='scalar fpdtype_t'
              ploc='in fpdtype_t[${str(ndims)}]'
              u='inout fpdtype_t[${str(nvars)}]'
              >
fpdtype_t src[${nvars}] = {};

% for mod, name in srcmacros:
    ${pyfr.expand(name, 't', 'u', 'ploc', 'src')};
% endfor

% for i in range(nvars):
    u[${i}] = src[${i}];
% endfor
</%pyfr:kernel>
