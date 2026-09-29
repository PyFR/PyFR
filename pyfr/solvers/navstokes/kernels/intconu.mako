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
<%
rmom = pyfr.matvec('rmat', 'ulin[{j} + 1]', ndims)
ulr = ['ulin[0]', *rmom, f'ulin[{nvars - 1}]']
%>
    fpdtype_t ulr[] = ${pyfr.carray(ulr)};
    % for i in range(nvars):
    urout[${i}] = ulr[${i}] - urin[${i}];
    % endfor
  % endif
% elif c['ldg-beta'] == 0.5:
  % if not rperiodic:
    % for i in range(nvars):
    ulout[${i}] = urin[${i}] - ulin[${i}];
    % endfor
  % else:
    // Rotate u_R into the LHS frame
<%
lmom = pyfr.matvec('rmat', 'urin[{j} + 1]', ndims, transpose=True)
url = ['urin[0]', *lmom, f'urin[{nvars - 1}]']
%>
    fpdtype_t url[] = ${pyfr.carray(url)};
    % for i in range(nvars):
    ulout[${i}] = url[${i}] - ulin[${i}];
    % endfor
  % endif
% elif not rperiodic:
  % for i in range(nvars):
    ulout[${i}] = ${0.5 + c['ldg-beta']}*(urin[${i}] - ulin[${i}]);
    urout[${i}] = ${0.5 - c['ldg-beta']}*(ulin[${i}] - urin[${i}]);
  % endfor
% else:
<%
lmom = pyfr.matvec('rmat', 'urin[{j} + 1]', ndims, transpose=True)
url = ['urin[0]', *lmom, f'urin[{nvars - 1}]']
%>
    fpdtype_t url[] = ${pyfr.carray(url)};
    fpdtype_t du[${nvars}];

    // Compute the common solution jumps in the LHS frame
  % for i in range(nvars):
    du[${i}] = url[${i}] - ulin[${i}];
    ulout[${i}] = ${0.5 + c['ldg-beta']}*du[${i}];
    du[${i}] *= ${c['ldg-beta'] - 0.5};
  % endfor

    // Rotate the RHS jump into the RHS frame
<%
rmom = pyfr.matvec('rmat', 'du[{j} + 1]', ndims)
urvals = ['du[0]', *rmom, f'du[{nvars - 1}]']
%>
  % for i in range(nvars):
    urout[${i}] = ${urvals[i]};
  % endfor
% endif
</%pyfr:kernel>
