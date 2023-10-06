<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.baseadvec.kernels.smats'/>
<%include file='pyfr.solvers.baseadvecdiff.kernels.artvisc'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<% smats = 'smats_l' if 'linear' in ktype else 'smats' %>

<%pyfr:kernel name='tflux' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              artvisc='in broadcast-col fpdtype_t'
              f='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              verts='in broadcast-col fpdtype_t[${str(nverts)}][${str(ndims)}]'
              upts='in broadcast-row fpdtype_t[${str(ndims)}]'>
% if 'linear' in ktype:
    // Compute the S matrices
    fpdtype_t ${smats}[${ndims}][${ndims}], djac;
    ${pyfr.expand('calc_smats_detj', 'verts', 'upts', smats, 'djac')};
% endif

    // Compute the flux (F = Fi + Fv)
    fpdtype_t ftemp[${ndims}][${nvars}];
    fpdtype_t p, v[${ndims}];
    ${pyfr.expand('inviscid_flux', 'u', 'ftemp', 'p', 'v')};
    ${pyfr.expand('viscous_flux_add', 'u', 'f', 'ftemp')};
    ${pyfr.expand('artificial_viscosity_add', 'f', 'ftemp', 'artvisc')};

    // Transform the fluxes
% for i, j in pyfr.ndrange(ndims, nvars):
    f[${i}][${j}] = ${' + '.join(f'{smats}[{i}][{k}]*ftemp[{k}][{j}]'
                                 for k in range(ndims))};
% endfor
</%pyfr:kernel>
