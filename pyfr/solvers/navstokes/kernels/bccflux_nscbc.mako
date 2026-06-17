<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>
<%include file='pyfr.solvers.euler.kernels.nscbc'/>
<%namespace file='pyfr.solvers.euler.kernels.nscbc' import='nscbc_body'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.${bctype}'/>

<%pyfr:macro name='total_flux' params='u, f, py:upt'>
  fpdtype_t p, v[${ndims}];
  fpdtype_t gradu[${ndims}][${nvars}];
  % for var in range(nvars):
    % for dim in range(ndims):
  gradu[${dim}][${var}] = gradu_upts[${dim*nupts + upt}][${var}];
    % endfor
  % endfor
  ${pyfr.expand('inviscid_flux', 'u', 'f', 'p', 'v')};
  ${pyfr.expand('viscous_flux_add', 'u', 'gradu', 'f')};
</%pyfr:macro>

<%pyfr:kernel name='bccflux_nscbc' ndim='1'
              u_upts='in view fpdtype_t[${str(nupts)}][${str(nvars)}]'
              u_fpts='inout view fpdtype_t[${str(nfpts)}][${str(nvars)}]'
              gradu_upts='in view fpdtype_t[${str(ndims*nupts)}][${str(nvars)}]'
              smats_upts='in fpdtype_t[${str(nupts)}][${str(ndims*ndims)}]'
              jacs_ffpts='in fpdtype_t[${str(nfacefpts)}]'
              m0='in broadcast fpdtype_t[${str(nfacefpts)}][${str(nupts)}]'
              m2='in broadcast fpdtype_t[${str(nfpts)}][${str(ndims*nupts)}]'
              m_div='in broadcast fpdtype_t[${str(nfacefpts)}][${str(ndims*nupts)}]'>
${nscbc_body()}
</%pyfr:kernel>
