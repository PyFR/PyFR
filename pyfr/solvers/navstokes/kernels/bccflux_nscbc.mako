<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>
<%include file='pyfr.solvers.navstokes.kernels.nscbc'/>
<%namespace file='pyfr.solvers.navstokes.kernels.nscbc' import='nscbc_body'/>

<%include file='pyfr.solvers.navstokes.kernels.bcs.${bctype}'/>

<%pyfr:kernel name='bccflux_nscbc' ndim='1'
              u_upts='in view fpdtype_t[${str(nupts)}][${str(nvars)}]'
              u_fpts='inout view fpdtype_t[${str(nfpts)}][${str(nvars)}]'
              gradu_upts='in view fpdtype_t[${str(ndims*nupts)}][${str(nvars)}]'
              smats_upts='in fpdtype_t[${str(nupts)}][${str(ndims*ndims)}]'
              jacs_ffpts='in fpdtype_t[${str(nfacefpts)}]'>
${nscbc_body()}
</%pyfr:kernel>
