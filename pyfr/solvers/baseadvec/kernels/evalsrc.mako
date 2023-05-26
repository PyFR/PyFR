<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
% for mod, name in srcmacros:
    <%include file='${mod}'/>
% endfor

<%pyfr:kernel name='evalsrc' ndim='2'
              t='scalar fpdtype_t'
              ploc='in fpdtype_t[${str(ndims)}]'
              u='inout fpdtype_t[${str(nvars)}]'
              >
fpdtype_t stemp[${nvars}] = {};

% for mod, name in srcmacros:
    ${pyfr.expand(name, 't', 'u', 'ploc', 'stemp')};
% endfor

% for i in range(nvars):
    u[${i}] = stemp[${i}];
% endfor
</%pyfr:kernel>