<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
% for mod, name in src_macros:
    <%include file='${mod}'/>
% endfor

<%
    # Write over the solution when the divergence was formed in the scratch
    if preserve_soln:
        dio, uio, out = 'in', 'inout', 'u'
    else:
        dio, uio, out = 'inout', 'in', 'divconf'

    # Scale the divergence where it is left transformed
    scale = 'rcpdjac*' if ktype == 'transformed' else ''
%>

<%pyfr:kernel name='negdivconf' ndim='2'
              t='scalar fpdtype_t'
              divconf='${dio} fpdtype_t[${str(nvars)}]'
              u='${uio} fpdtype_t[${str(nvars)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              rcpdjac='in fpdtype_t'>
fpdtype_t src[${nvars}] = {};

% for mod, name in src_macros:
    ${pyfr.expand(name, 't', 'u', 'ploc', 'src')};
% endfor

% for i in range(nvars):
    ${out}[${i}] = -${scale}divconf[${i}] + src[${i}];
% endfor
</%pyfr:kernel>
