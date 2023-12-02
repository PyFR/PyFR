<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.aceuler.kernels.flux'/>
<%include file='pyfr.solvers.acnavstokes.kernels.flux'/>
<%include file='pyfr.solvers.baseadvec.kernels.smats'/>
<%include file='pyfr.solvers.baseadvecdiff.kernels.transform_grad'/>

<% smats = 'smats_l' if 'linear' in ktype else 'smats' %>

<% gradu = 'gradu' if 'fused' in ktype else 'f' %>
<% smats = 'smats_l' if 'linear' in ktype else 'smats' %>
<% rcpdjac = 'rcpdjac_l' if 'linear' in ktype else 'rcpdjac' %>

<%pyfr:kernel name='tflux' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              f='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'
              verts='in broadcast-col fpdtype_t[${str(nverts)}][${str(ndims)}]'
              upts='in broadcast-row fpdtype_t[${str(ndims)}]'>
% if 'linear' in ktype:
    // Compute the S matrices
    fpdtype_t ${smats}[${ndims}][${ndims}], djac;
    ${pyfr.expand('calc_smats_detj', 'verts', 'upts', smats, 'djac')};
    fpdtype_t ${rcpdjac} = 1 / djac;
% endif

% if 'fused' in ktype:
    // Transform the corrected gradient
    ${pyfr.expand('transform_grad', gradu, smats, rcpdjac)};
% endif

    // Compute the flux (F = Fi + Fv)
    fpdtype_t ftemp[${ndims}][${nvars}];
    ${pyfr.expand('inviscid_flux', 'u', 'ftemp')};
    ${pyfr.expand('viscous_flux_add', 'u', gradu, 'ftemp')};

    // Transform the fluxes
% for i, j in pyfr.ndrange(ndims, nvars):
    f[${i}][${j}] = ${' + '.join(f'{smats}[{i}][{k}]*ftemp[{k}][{j}]'
                                 for k in range(ndims))};
% endfor
</%pyfr:kernel>
