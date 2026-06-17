<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%!
def _eigvecs(nvars, ndims):
    s2 = 2**-0.5
    WU    = [[0]*nvars for _ in range(nvars)]   # Phi = WU . div
    WUinv = [[0]*nvars for _ in range(nvars)]   # div = WUinv . Phi

    # Entropy wave
    WU[0][0], WU[0][nvars-1] = '1.0 - gmo*invcsq*k', '-gmo*invcsq'
    WUinv[0][0], WUinv[nvars-1][0] = '1.0', 'k'
    for i in range(ndims):
        WU[0][i+1], WUinv[i+1][0] = f'gmo*invcsq*v[{i}]', f'v[{i}]'

    # Vortical waves (one per tangent)
    for j in range(ndims-1):
        WU[j+1][0], WUinv[nvars-1][j+1] = f'-invrho*ut{j+1}', f'rho*ut{j+1}'
        for i in range(ndims):
            WU[j+1][i+1]    = f'invrho*t{j+1}[{i}]'
            WUinv[i+1][j+1] = f'rho*t{j+1}[{i}]'

    # Acoustic waves (+ uses +c, - uses -c; the density-row un term flips)
    for w, sn in zip((nvars-2, nvars-1), ('+', '-')):
        cs = '' if sn == '+' else '-'
        us = '-' if sn == '+' else '+'
        WU[w][0]       = f'{s2}*invc*invrho*(gmo*k {us} c*un)'
        WU[w][nvars-1] = f'{s2}*invc*invrho*gmo'
        WUinv[0][w]       = f'{s2}*rho*invc'
        WUinv[nvars-1][w] = f'{s2}*rho*(c/gmo + k*invc {sn} un)'
        for i in range(ndims):
            WU[w][i+1]    = f'{s2}*invc*invrho*({cs}c*norm_nl[{i}] - gmo*v[{i}])'
            WUinv[i+1][w] = f'{s2}*rho*invc*({cs}c*norm_nl[{i}] + v[{i}])'
    return WU, WUinv


def _wave_idx(names, nvars, ndims):
    groups = {'entropy': [0], 'vortical': list(range(1, ndims)),
              'acoustic+': [nvars-2], 'acoustic-': [nvars-1]}
    return [i for nm in names for i in groups[nm]]
%>


## Discontinuous normal flux at all NSCBC face flux points
<%pyfr:macro name='disc_nflux' params='tf, nf'>
for (int fpt = 0; fpt < ${nfpts}; fpt++)
  for (int upt = 0; upt < ${nupts}; upt++)
  {
  % for comp in range(ndims):
  % for var in range(nvars):
    nf[fpt][${var}] += tf[upt][${comp}][${var}]*m2[fpt][${comp*nupts} + upt];
  % endfor
  % endfor
  }
</%pyfr:macro>


## Physical normal and face coordinate system at face f
<%pyfr:macro name='face_cs' params='smats, norm_nl, t1, t2, py:f'>
fpdtype_t smats_fpt[${ndims}][${ndims}] = {{0}};
for (int upt = 0; upt < ${nupts}; upt++)
{
  % for comp in range(ndims):
  % for phys in range(ndims):
  smats_fpt[${comp}][${phys}] += m0[${f}][upt]*smats[upt][${comp*ndims + phys}];
  % endfor
  % endfor
}

% for phys in range(ndims):
norm_nl[${phys}] = 0.0;
  % for comp in range(ndims):
    % if abs(norm_ref[f,comp]) > 0.0:
norm_nl[${phys}] += smats_fpt[${comp}][${phys}] * ${norm_ref[f,comp]};
    % endif
  % endfor
% endfor

fpdtype_t inv_mag = 1.0/sqrt(${pyfr.dot('norm_nl[{i}]', i=ndims)});
% for phys in range(ndims):
norm_nl[${phys}] *= inv_mag;
% endfor

% if ndims == 2:
t1[0] = -norm_nl[1];
t1[1] = norm_nl[0];
% elif ndims == 3:
fpdtype_t sign = copysign(1.0, norm_nl[2]);
fpdtype_t a = -1.0/(sign + norm_nl[2]);
fpdtype_t b = norm_nl[0]*norm_nl[1]*a;
t1[0] = 1.0 + sign*norm_nl[0]*norm_nl[0]*a;
t1[1] = sign*b;
t1[2] = -sign*norm_nl[0];
t2[0] = b;
t2[1] = sign + norm_nl[1]*norm_nl[1]*a;
t2[2] = -norm_nl[1];
% endif
</%pyfr:macro>


## Reference-space flux divergence at face f
<%pyfr:macro name='ref_div' params='tf, div, py:f'>
% for var in range(nvars):
div[${var}] = 0.0;
% endfor
for (int upt = 0; upt < ${nupts}; upt++)
{
  % for dim in range(ndims):
  % for var in range(nvars):
  div[${var}] += tf[upt][${dim}][${var}]*m_div[${f}][${dim*nupts} + upt];
  % endfor
  % endfor
}
</%pyfr:macro>


## Apply the prescribed change to the characteristic waves named in `waves`
## and project it back onto the conserved divergence.  Only the modified
## waves are projected; the rest cancel exactly in the difference.
<%pyfr:macro name='project_waves' params='div, Phi, dd, u, p, v, norm_nl, t1, t2, py:f, py:waves'>
<%
    WU, WUinv = _eigvecs(nvars, ndims)
    widx = _wave_idx(waves, nvars, ndims)
%>
  fpdtype_t rho = u[0], invrho = 1.0/rho;
  fpdtype_t c = sqrt(${c['gamma']}*p*invrho), invc = 1.0/c, invcsq = invc*invc;
  fpdtype_t gmo = ${c['gamma'] - 1.0};
  fpdtype_t k = 0.5*(${pyfr.dot('v[{i}]', i=ndims)});
  fpdtype_t un = ${pyfr.dot('norm_nl[{i}]', 'v[{i}]', i=ndims)};
  fpdtype_t ut1 = ${pyfr.dot('t1[{i}]', 'v[{i}]', i=ndims)};
% if ndims == 3:
  fpdtype_t ut2 = ${pyfr.dot('t2[{i}]', 'v[{i}]', i=ndims)};
% endif

  ## Change in each prescribed wave: target amplitude minus projected one
% for w in widx:
  fpdtype_t dphi${w} = Phi[${w}]${''.join(f' - ({WU[w][j]})*div[{j}]' for j in range(nvars) if WU[w][j])};
% endfor

  ## Scatter the wave changes back onto the conserved divergence
% for i in range(nvars):
  dd[${f}][${i}] = ${' + '.join(f'({WUinv[i][w]})*dphi{w}' for w in widx if WUinv[i][w]) or '0.0'};
% endfor
</%pyfr:macro>


## FR flux correction
<%pyfr:macro name='fr_update' params='uf, nf, dd'>
% for f, fpt_idx in enumerate(facefpts):
  % for var in range(nvars):
  {
    fpdtype_t R = 0.0;
    % for j, intfpt_idx in enumerate(intfpts):
      % if abs(GB_inv_GI[f,j]) > 0.0:
      R += ${GB_inv_GI[f,j]}*(uf[${intfpt_idx}][${var}] - nf[${intfpt_idx}][${var}]);
      % endif
    % endfor
    fpdtype_t A = 0.0;
    % for j in range(nfacefpts):
      % if abs(GB_inv[f,j]) > 0.0:
      A += ${GB_inv[f,j]} * dd[${j}][${var}];
      % endif
    % endfor

    uf[${fpt_idx}][${var}] = nf[${fpt_idx}][${var}] + A - R;
  }
  % endfor
% endfor
</%pyfr:macro>


## NSCBC boundary algorithm.  The reference flux at each solution point is
## supplied by a system-local `total_flux` macro, so this body is shared
## unchanged between the Euler and Navier-Stokes systems.
<%def name='nscbc_body()'>
## Transformed flux at the solution points
fpdtype_t tf_upts[${nupts}][${ndims}][${nvars}] = {{{0}}};
% for upt in range(nupts):
{
  fpdtype_t u[${nvars}];
  % for var in range(nvars):
  u[${var}] = u_upts[${upt}][${var}];
  % endfor

  fpdtype_t f[${ndims}][${nvars}];
  ${pyfr.expand('total_flux', 'u', 'f', upt)}
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

  ## Prescribe the modified characteristic waves, then project the change
  ## back onto the conserved divergence
  fpdtype_t Phi[${nvars}];
  ${pyfr.expand('compute_wave_amp', 'ul', 'p', 'v', 'norm_nl', 't1', 't2', 'Phi', 'jac')};
  ${pyfr.expand('project_waves', 'div', 'Phi', 'delta_div', 'ul', 'p', 'v', 'norm_nl', 't1', 't2', f, waves)}
}
% endfor

${pyfr.expand('fr_update', 'u_fpts', 'tnf_D', 'delta_div')}
</%def>
