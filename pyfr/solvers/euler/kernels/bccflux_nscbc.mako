<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>
<%include file='pyfr.solvers.euler.kernels.nscbc'/>
<%namespace file='pyfr.solvers.euler.kernels.nscbc' import='nscbc_body'/>
<%include file='pyfr.solvers.euler.kernels.bcs.${bctype}'/>

<%pyfr:macro name='total_flux' params='u, f, py:upt'>
  fpdtype_t p, v[${ndims}];
  ${pyfr.expand('inviscid_flux', 'u', 'f', 'p', 'v')};
</%pyfr:macro>

<%pyfr:kernel name='bccflux_nscbc' ndim='1'
              u_upts='in view fpdtype_t[${str(nupts)}][${str(nvars)}]'
              u_fpts='inout view fpdtype_t[${str(nfpts)}][${str(nvars)}]'
              smats_upts='in fpdtype_t[${str(nupts)}][${str(ndims*ndims)}]'
              jacs_ffpts='in fpdtype_t[${str(nfacefpts)}]'
              m0='in broadcast fpdtype_t[${str(nfacefpts)}][${str(nupts)}]'
              m2='in broadcast fpdtype_t[${str(nfpts)}][${str(ndims*nupts)}]'
              m_div='in broadcast fpdtype_t[${str(nfacefpts)}][${str(ndims*nupts)}]'>
${nscbc_body()}
</%pyfr:kernel>
