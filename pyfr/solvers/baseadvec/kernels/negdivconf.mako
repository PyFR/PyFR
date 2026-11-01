<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
% for mod, name in src_macros:
    <%include file='${mod}'/>
% endfor

<%
    # Write over the solution when the divergence was formed in the scratch
    if preserve_soln:
        kargs = {'tdivtconf': f'in fpdtype_t[{nvars}]',
                 'u': f'inout fpdtype_t[{nvars}]'}
        out = 'u'
    else:
        kargs = {'tdivtconf': f'inout fpdtype_t[{nvars}]',
                 'u': f'in fpdtype_t[{nvars}]'}
        out = 'tdivtconf'
%>

<%pyfr:kernel name='negdivconf' ndim='2' kargs='${kargs}'
              t='scalar fpdtype_t'
              ploc='in fpdtype_t[${str(ndims)}]'
              rcpdjac='in fpdtype_t'>
fpdtype_t src[${nvars}] = {};

% for mod, name in src_macros:
    ${pyfr.expand(name, 't', 'u', 'ploc', 'src')};
% endfor

% for i in range(nvars):
    ${out}[${i}] = -rcpdjac*tdivtconf[${i}] + src[${i}];
% endfor
</%pyfr:kernel>
