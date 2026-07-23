<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.baseadvec.kernels.smats'/>

<% smats = 'smats_l' if 'linear' in ktype else 'smats' %>
<% rcpdjac_v = 'rcpdjac_l' if 'linear' in ktype else 'rcpdjac' %>

<%pyfr:kernel name='wavespeed' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'
              verts='in broadcast-col fpdtype_t[${str(nverts)}][${str(ndims)}]'
              upts='in broadcast-row fpdtype_t[${str(ndims)}]'
              wspd='out broadcast-col reduce(max) fpdtype_t'>
% if 'linear' in ktype:
    fpdtype_t ${smats}[${ndims}][${ndims}], djac;
    ${pyfr.expand('calc_smats_detj', 'verts', 'upts', smats, 'djac')};
    fpdtype_t ${rcpdjac_v} = 1/djac;
% endif

    // Compute the primitive state
    ${fluid.decl('u', 'v, p, a')}

    fpdtype_t lam = 0;
% for i in range(ndims):
    lam += fabs(${' + '.join(f'({smats}[{i}][{j}]*{rcpdjac_v})*v[{j}]'
                             for j in range(ndims))})
         + a*sqrt(${' + '.join(f'({smats}[{i}][{j}]*{rcpdjac_v})'
                               f'*({smats}[{i}][{j}]*{rcpdjac_v})'
                               for j in range(ndims))});
% endfor

    wspd = lam;
</%pyfr:kernel>
