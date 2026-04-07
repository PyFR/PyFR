<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='intcent' ndim='1'
              entmin_lhs='inout view fpdtype_t'
              entmin_rhs='inout view fpdtype_t'>
    entmin_lhs = entmin_rhs = fmin(entmin_lhs, entmin_rhs);
</%pyfr:kernel>
