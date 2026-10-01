<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.baseadvec.kernels.transform'/>

<% beta = c['ldg-beta'] %>

<%pyfr:kernel name='intconu' ndim='1'
              ulin='in view fpdtype_t[${str(nvars)}]'
              urin='in view fpdtype_t[${str(nvars)}]'
              ulout='out view fpdtype_t[${str(nvars)}]'
              urout='out view fpdtype_t[${str(nvars)}]'
              rmat='in broadcast fpdtype_t[${str(ndims)}][${str(ndims)}]'>
    fpdtype_t ur[] = ${pyfr.array('urin[{i}]', i=nvars)};
% if rperiodic:
    // Rotate u_R into the LHS frame
    ${pyfr.expand('rotate', 'rmat', 'ur', off=1, transpose=True)};
% endif

    // Compute the common solution jumps in the LHS frame
    fpdtype_t du[] = ${pyfr.array('ur[{i}] - ulin[{i}]', i=nvars)};

% if beta != -0.5:
% for i in range(nvars):
    ulout[${i}] = ${0.5 + beta}*du[${i}];
% endfor
% endif

% if rperiodic and beta != 0.5:
    // Rotate the RHS jump into the RHS frame
    ${pyfr.expand('rotate', 'rmat', 'du', off=1, transpose=False)};
% endif

% if beta != 0.5:
% for i in range(nvars):
    urout[${i}] = ${beta - 0.5}*du[${i}];
% endfor
% endif
</%pyfr:kernel>
