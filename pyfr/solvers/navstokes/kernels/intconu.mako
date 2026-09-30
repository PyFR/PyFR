<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='intconu' ndim='1'
              ulin='in view fpdtype_t[${str(nvars)}]'
              urin='in view fpdtype_t[${str(nvars)}]'
              ulout='out view fpdtype_t[${str(nvars)}]'
              urout='out view fpdtype_t[${str(nvars)}]'
              rmat='in broadcast fpdtype_t[${str(ndims)}][${str(ndims)}]'>
% if c['ldg-beta'] == -0.5:
  % if not rperiodic:
    % for i in range(nvars):
    urout[${i}] = ulin[${i}] - urin[${i}];
    % endfor
  % else:
    // Rotate u_L into the RHS frame
<% ulr = pyfr.rotstate('rmat', 'ulin', ndims) %>
    % for i, expr in enumerate(ulr):
    urout[${i}] = ${expr} - urin[${i}];
    % endfor
  % endif
% elif c['ldg-beta'] == 0.5:
  % if not rperiodic:
    % for i in range(nvars):
    ulout[${i}] = urin[${i}] - ulin[${i}];
    % endfor
  % else:
    // Rotate u_R into the LHS frame
<% url = pyfr.rotstate('rmat', 'urin', ndims, True) %>
    % for i, expr in enumerate(url):
    ulout[${i}] = ${expr} - ulin[${i}];
    % endfor
  % endif
% elif not rperiodic:
  % for i in range(nvars):
    ulout[${i}] = ${0.5 + c['ldg-beta']}*(urin[${i}] - ulin[${i}]);
    urout[${i}] = ${0.5 - c['ldg-beta']}*(ulin[${i}] - urin[${i}]);
  % endfor
% else:
    fpdtype_t du[${nvars}];

    // Compute the common solution jumps in the LHS frame
<% url = pyfr.rotstate('rmat', 'urin', ndims, True) %>
  % for i, expr in enumerate(url):
    du[${i}] = ${expr} - ulin[${i}];
    ulout[${i}] = ${0.5 + c['ldg-beta']}*du[${i}];
    du[${i}] *= ${c['ldg-beta'] - 0.5};
  % endfor

    // Rotate the RHS jump into the RHS frame
<% urvals = pyfr.rotstate('rmat', 'du', ndims) %>
  % for i, expr in enumerate(urvals):
    urout[${i}] = ${expr};
  % endfor
% endif
</%pyfr:kernel>
