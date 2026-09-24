<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='mpiconu' ndim='1'
              ulin='in view fpdtype_t[${str(nvars)}]'
              urin='in mpi fpdtype_t[${str(nvars)}]'
              ulout='out view fpdtype_t[${str(nvars)}]'>
% for i in range(nvars):
% if c['ldg-beta'] == 0.5:
    ulout[${i}] = urin[${i}] - ulin[${i}];
% else:
    ulout[${i}] = ${0.5 + c['ldg-beta']}*(urin[${i}] - ulin[${i}]);
% endif
% endfor
</%pyfr:kernel>

