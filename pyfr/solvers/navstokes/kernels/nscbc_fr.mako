<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

## Discontinuous normal flux at all NSCBC face flux points
<%pyfr:macro name='disc_nflux' params='tf, nf'>
% for fpt_idx in fptidx:
{
  % for upt in range(nupts):
  % for var in range(nvars):
  % for comp in range(ndims):
    % if abs(m2[fpt_idx,comp,upt]) > 0.0:
    nf[${fpt_idx}][${var}] += tf[${upt}][${comp}][${var}]*${m2[fpt_idx,comp,upt]};
    % endif
  % endfor
  % endfor
  % endfor
}
% endfor
</%pyfr:macro>


## Physical normal and face coordinate system at face f
<%pyfr:macro name='face_cs' params='smats, norm_nl, t1, t2, py:f'>
fpdtype_t smats_fpt[${ndims}][${ndims}] = {{0}};
% for comp in range(ndims):
  % for phys in range(ndims):
    % for upt in range(nupts):
      % if abs(m0[f,upt]) > 0.0:
smats_fpt[${comp}][${phys}] += ${m0[f,upt]} * smats[${upt}][${comp*ndims + phys}];
      % endif
    % endfor
  % endfor
% endfor

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
  % for dim in range(ndims):
    % for upt in range(nupts):
      % if abs(m12[f,dim,upt]) > 0.0:
div[${var}] += tf[${upt}][${dim}][${var}] * ${m12[f,dim,upt]};
      % endif
    % endfor
  % endfor
% endfor
</%pyfr:macro>


## FR flux correction
<%pyfr:macro name='fr_update' params='uf, nf, dd'>
% for f, fpt_idx in enumerate(facefpts):
{
  % for var in range(nvars):
  {
    fpdtype_t R = 0;
    % for j, intfpt_idx in enumerate(intfpts):
      % if abs(GB_inv_GI[f,j]) > 0.0:
      R += ${GB_inv_GI[f,j]}*(uf[${intfpt_idx}][${var}] - nf[${intfpt_idx}][${var}]);
      % endif
    % endfor
    fpdtype_t A = 0.0;
    % for j, fpt_jdx in enumerate(facefpts):
      % if abs(GB_inv[f,j]) > 0.0:
      A += ${GB_inv[f,j]} * dd[${j}][${var}];
      % endif
    % endfor

    uf[${fpt_idx}][${var}] = nf[${fpt_idx}][${var}] + A - R;
  }
  % endfor
}
% endfor
</%pyfr:macro>
