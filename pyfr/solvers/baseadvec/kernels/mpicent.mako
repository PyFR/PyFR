<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='mpicent' ndim='1'
              entmin_lhs='inout view fpdtype_t'
              entmin_rhs='in mpi fpdtype_t'>
    entmin_lhs = fmin(entmin_lhs, entmin_rhs);
</%pyfr:kernel>
