<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%pyfr:kernel name='vtxreduce' ndim='1'
              vtx='out view reduce(max) fpdtype_t'
              recv='in fpdtype_t'>
    vtx = recv;
</%pyfr:kernel>
