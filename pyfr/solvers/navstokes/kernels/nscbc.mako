<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.nscbc_fr'/>
<%include file='pyfr.solvers.navstokes.kernels.nscbc_decomp'/>

<%def name="nscbc_body()">
## Compute transformed flux at solution points
fpdtype_t tf_upts[${nupts}][${ndims}][${nvars}] = {{{0}}};
% for upt in range(nupts):
{
  fpdtype_t u[${nvars}];
  % for var in range(nvars):
  u[${var}] = u_upts[${upt}][${var}];
  % endfor
  fpdtype_t gradu[${ndims}][${nvars}];
  % for var in range(nvars):
    % for dim in range(ndims):
  gradu[${dim}][${var}] = gradu_upts[${dim*nupts + upt}][${var}];
    % endfor
  % endfor

  fpdtype_t f[${ndims}][${nvars}];
  fpdtype_t p, v[${ndims}];
  ${pyfr.expand('inviscid_flux', 'u', 'f', 'p', 'v')};
  ${pyfr.expand('viscous_flux_add', 'u', 'gradu', 'f')};
  % for var in range(nvars):
    % for comp in range(ndims):
      % for phys in range(ndims):
  tf_upts[${upt}][${comp}][${var}] += smats_upts[${upt}][${comp*ndims + phys}]*f[${phys}][${var}];
      % endfor
    % endfor
  % endfor
}
% endfor

fpdtype_t tnf_D[${nfpts}][${nvars}] = {{0}};
${pyfr.expand('disc_nflux', 'tf_upts', 'tnf_D')}

fpdtype_t delta_div[${nfacefpts}][${nvars}];
% for f, fpt_idx in enumerate(facefpts):
{
  fpdtype_t jac = jacs_ffpts[${f}];

  fpdtype_t norm_nl[${ndims}];
  fpdtype_t t1[${ndims}];
% if ndims == 3:
  fpdtype_t t2[${ndims}];
% endif
  ${pyfr.expand('face_cs', 'smats_upts', 'norm_nl', 't1', 't2', f)}

  ## Load state at flux point and compute primitives
  fpdtype_t ul[${nvars}];
  % for var in range(nvars):
  ul[${var}] = u_fpts[${fpt_idx}][${var}];
  % endfor
  fpdtype_t invrho = 1.0/ul[0];
  fpdtype_t v[${ndims}];
  % for i in range(ndims):
  v[${i}] = ul[${i+1}]*invrho;
  % endfor
  fpdtype_t p = ${c['gamma'] - 1}*(ul[${nvars-1}] - 0.5*ul[0]*(${pyfr.dot('v[{i}]', i=ndims)}));

  fpdtype_t div[${nvars}];
  ${pyfr.expand('ref_div', 'tf_upts', 'div', f)}

  ## Project divergence onto characteristics, apply BC, project back
  fpdtype_t Phi[${nvars}];
  ${pyfr.expand(f'WU_dot_div-{decomp_type}','div','Phi','ul','p','v')}

  ${pyfr.expand('compute_wave_amp', 'ul', 'p', 'v', 'Phi', 'jac')};

  fpdtype_t div_star[${nvars}];
  ${pyfr.expand(f'WUinv_dot_Phi-{decomp_type}','Phi','div_star','ul','p','v')};

  % for var in range(nvars):
  delta_div[${f}][${var}] = div_star[${var}] - div[${var}];
  % endfor
}
% endfor

${pyfr.expand('fr_update', 'u_fpts', 'tnf_D', 'delta_div')}

</%def>
