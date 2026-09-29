<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<% vmap = (None, None, 1) %>

<%pyfr:kernel name='intconu' ndim='1'
              ulin='in view fpdtype_t[${str(nvars)}]'
              urin='in view fpdtype_t[${str(nvars)}]'
              ulout='out view fpdtype_t[${str(nvars)}]'
              urout='out view fpdtype_t[${str(nvars)}]'>
% if c['ldg-beta'] == -0.5:
  % if rot is None:
    % for i in range(nvars):
    urout[${i}] = ulin[${i}] - urin[${i}];
    % endfor
  % else:
    // Rotate u_L into the RHS frame
    fpdtype_t ulr[${nvars}];
    ulr[0] = ulin[0];
    ulr[${nvars - 1}] = ulin[${nvars - 1}];
    ${pyfr.constmatvec(rot, 'ulr', 'ulin', vmap, vmap)}
    % for i in range(nvars):
    urout[${i}] = ulr[${i}] - urin[${i}];
    % endfor
  % endif
% elif c['ldg-beta'] == 0.5:
  % if rot is None:
    % for i in range(nvars):
    ulout[${i}] = urin[${i}] - ulin[${i}];
    % endfor
  % else:
    // Rotate u_R into the LHS frame
    fpdtype_t url[${nvars}];
    url[0] = urin[0];
    url[${nvars - 1}] = urin[${nvars - 1}];
    ${pyfr.constmatvec(rot.T, 'url', 'urin', vmap, vmap)}
    % for i in range(nvars):
    ulout[${i}] = url[${i}] - ulin[${i}];
    % endfor
  % endif
% elif rot is None:
  % for i in range(nvars):
    ulout[${i}] = ${0.5 + c['ldg-beta']}*(urin[${i}] - ulin[${i}]);
    urout[${i}] = ${0.5 - c['ldg-beta']}*(ulin[${i}] - urin[${i}]);
  % endfor
% else:
    fpdtype_t url[${nvars}], du[${nvars}];
    url[0] = urin[0];
    url[${nvars - 1}] = urin[${nvars - 1}];
    ${pyfr.constmatvec(rot.T, 'url', 'urin', vmap, vmap)}

    // Compute the common solution jumps in the LHS frame
  % for i in range(nvars):
    du[${i}] = url[${i}] - ulin[${i}];
    ulout[${i}] = ${0.5 + c['ldg-beta']}*du[${i}];
    du[${i}] *= ${c['ldg-beta'] - 0.5};
  % endfor

    // Rotate the RHS jump into the RHS frame
    urout[0] = du[0];
    urout[${nvars - 1}] = du[${nvars - 1}];
    ${pyfr.constmatvec(rot, 'urout', 'du', vmap, vmap)}
% endif
</%pyfr:kernel>
