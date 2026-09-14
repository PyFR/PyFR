<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.periodic'/>

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
    ${pyfr.expand('rotate_from_lhs', 'ulr', 'ulin', rot)};
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
    ${pyfr.expand('rotate_into_lhs', 'url', 'urin', rot)};
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
    ${pyfr.expand('rotate_into_lhs', 'url', 'urin', rot)};

    // Compute the common solution jumps in the LHS frame
  % for i in range(nvars):
    du[${i}] = url[${i}] - ulin[${i}];
    ulout[${i}] = ${0.5 + c['ldg-beta']}*du[${i}];
    du[${i}] *= ${c['ldg-beta'] - 0.5};
  % endfor

    // Rotate the RHS jump into the RHS frame
    ${pyfr.expand('rotate_from_lhs', 'urout', 'du', rot)};
% endif
</%pyfr:kernel>
