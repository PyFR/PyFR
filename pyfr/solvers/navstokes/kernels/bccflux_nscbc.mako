<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<%include file='pyfr.solvers.navstokes.kernels.bcs.${bctype}'/>

<% sq2 = 2**0.5 %>
<% invsq2 = 2**-0.5 %>

<%def name="nidx(comp,phys)">
  <% return comp*ndims + phys %>
</%def>\
<%def name="nfidx(dim, fpt)">
  <% return dim*nfpts + fpt %>
</%def>\
<%def name="nuidx(dim, upt)">
  <% return dim*nupts + upt %>
</%def>\

<%pyfr:macro name='WU_dot_div-cartesian' params='div, Phi, u, p, v'>
  fpdtype_t rho = u[0];
  fpdtype_t invrho = 1.0/rho;
  fpdtype_t c = sqrt(${c['gamma']}*p*invrho);
  fpdtype_t invc = 1.0/c;
  fpdtype_t invcsq = invc*invc;

  fpdtype_t nx = norm_nl[0];
  fpdtype_t ny = norm_nl[1];

  fpdtype_t gmo = ${c['gamma'] - 1.0};

  fpdtype_t k = 0.5*(${pyfr.dot('v[{i}]', i=ndims)});

% if ndims == 2:

  Phi[0] = div[0] - gmo*invcsq*(div[0]*k - div[1]*v[0] - div[2]*v[1] + div[3]);
  Phi[1] = invrho*(div[0]*(ny*v[0] - nx*v[1]) - div[1]*ny + div[2]*nx);
  Phi[2] = ${invsq2}*invc*invrho*(gmo*(div[3] + div[0]*k - (div[1]*v[0] + div[2]*v[1])) + c*(div[1]*nx + div[2]*ny - div[0]*(nx*v[0] + ny*v[1])));
  Phi[3] = ${invsq2}*invc*invrho*(gmo*(div[3] + div[0]*k - (div[1]*v[0] + div[2]*v[1])) - c*(div[1]*nx + div[2]*ny - div[0]*(nx*v[0] + ny*v[1])));

% elif ndims == 3:

  fpdtype_t nz = norm_nl[2];

  Phi[0] = gmo*nx*invcsq*(-div[4] - div[0]*k + div[1]*v[0] + div[2]*v[1] + div[3]*v[2]) + invrho*(-div[3]*ny + div[2]*nz + div[0]*(nx*rho - nz*v[1] + ny*v[2]));
  Phi[1] = gmo*ny*invcsq*(-div[4] - div[0]*k + div[1]*v[0] + div[2]*v[1] + div[3]*v[2]) + invrho*( div[3]*nx - div[1]*nz + div[0]*(ny*rho + nz*v[0] - nx*v[2]));
  Phi[2] = gmo*nz*invcsq*(-div[4] - div[0]*k + div[1]*v[0] + div[2]*v[1] + div[3]*v[2]) + invrho*(-div[2]*nx + div[1]*ny + div[0]*(nz*rho - ny*v[0] + nx*v[1]));
  Phi[3] = invc*invrho*${invsq2}*(gmo*(div[4] + div[0]*k - (div[1]*v[0] + div[2]*v[1] + div[3]*v[2])) + c*(div[1]*nx + div[2]*ny + div[3]*nz - div[0]*(nx*v[0] + ny*v[1] + nz*v[2])));
  Phi[4] = invc*invrho*${invsq2}*(gmo*(div[4] + div[0]*k - (div[1]*v[0] + div[2]*v[1] + div[3]*v[2])) - c*(div[1]*nx + div[2]*ny + div[3]*nz - div[0]*(nx*v[0] + ny*v[1] + nz*v[2])));

% endif
</%pyfr:macro>


<%pyfr:macro name='WUinv_dot_Phi-cartesian' params='Phi, div, u, p, v'>
  fpdtype_t rho = u[0];
  fpdtype_t invrho = 1.0/rho;
  fpdtype_t c = sqrt(${c['gamma']}*p*invrho);
  fpdtype_t invc = 1.0/c;

  fpdtype_t nx = norm_nl[0];
  fpdtype_t ny = norm_nl[1];

  fpdtype_t gmo = ${c['gamma'] - 1.0};

  fpdtype_t k = 0.5*(${pyfr.dot('v[{i}]', i=ndims)});

  % if ndims == 2:

  div[0] = Phi[0] + ${invsq2}*rho*invc*(Phi[2] + Phi[3]);
  div[1] = -Phi[1]*ny*rho + Phi[0]*v[0] + ${invsq2}*rho*invc*(Phi[2]*(c*nx + v[0]) - Phi[3]*(c*nx - v[0]));
  div[2] =  Phi[1]*nx*rho + Phi[0]*v[1] + ${invsq2}*rho*invc*(Phi[2]*(c*ny + v[1]) - Phi[3]*(c*ny - v[1]));
  div[3] = k*Phi[0] - Phi[1]*rho*(ny*v[0] - nx*v[1]) + ${invsq2}*rho*((Phi[3] + Phi[2])*(c/gmo + k*invc) + (Phi[2] - Phi[3])*(nx*v[0] + ny*v[1]));

  % elif ndims == 3:
  fpdtype_t nz = norm_nl[2];

  div[0] = Phi[0]*nx + Phi[1]*ny + Phi[2]*nz + ${invsq2}*invc*rho*(Phi[3] + Phi[4]);
  div[1] = rho*(Phi[2]*ny - Phi[1]*nz) + Phi[0]*nx*v[0] + Phi[1]*ny*v[0] + Phi[2]*nz*v[0] + ${invsq2}*invc*rho*(Phi[3]*(c*nx+v[0]) + Phi[4]*(-c*nx + v[0]));
  div[2] = rho*(Phi[0]*nz - Phi[2]*nx) + Phi[0]*nx*v[1] + Phi[1]*ny*v[1] + Phi[2]*nz*v[1] + ${invsq2}*invc*rho*(Phi[3]*(c*ny+v[1]) + Phi[4]*(-c*ny + v[1]));
  div[3] = rho*(Phi[1]*nx - Phi[0]*ny) + Phi[0]*nx*v[2] + Phi[1]*ny*v[2] + Phi[2]*nz*v[2] + ${invsq2}*invc*rho*(Phi[3]*(c*nz+v[2]) + Phi[4]*(-c*nz + v[2]));
  div[4] = Phi[0]*(k*nx + nz*rho*v[1] - ny*rho*v[2]) +
          Phi[1]*(k*ny - nz*rho*v[0] + nx*rho*v[2]) +
          Phi[2]*(k*nz + ny*rho*v[0] - nx*rho*v[1]) +
        rho*invc*${invsq2}/gmo*(Phi[3]*(c*c + gmo*k + c*gmo*(nx*v[0] + ny*v[1] + nz*v[2]))
                              + Phi[4]*(c*c + gmo*k - c*gmo*(nx*v[0] + ny*v[1] + nz*v[2])));

  % endif
</%pyfr:macro>


<%pyfr:macro name='WU_dot_div-normal' params='div, Phi, u, p, v'>
  fpdtype_t rho = u[0];
  fpdtype_t invrho = 1.0/rho;
  fpdtype_t c = sqrt(${c['gamma']}*p*invrho);
  fpdtype_t invc = 1.0/c;
  fpdtype_t invcsq = invc*invc;
  fpdtype_t gmo = ${c['gamma'] - 1.0};
  fpdtype_t k = 0.5*(${pyfr.dot('v[{i}]', i=ndims)});

  fpdtype_t un = ${pyfr.dot('norm_nl[{i}]','v[{i}]', i=ndims)};
  fpdtype_t ut1 = ${pyfr.dot('t1[{i}]','v[{i}]', i=ndims)};
  % if ndims == 3:
  fpdtype_t ut2 = ${pyfr.dot('t2[{i}]','v[{i}]', i=ndims)};
  % endif

  fpdtype_t div_v = ${pyfr.dot('div[{i} + 1]','v[{i}]', i=ndims)};
  fpdtype_t div_norm = ${pyfr.dot('div[{i} + 1]','norm_nl[{i}]', i=ndims)};
  fpdtype_t div_t1 = ${pyfr.dot('div[{i} + 1]','t1[{i}]', i=ndims)};
  % if ndims == 3:
  fpdtype_t div_t2 = ${pyfr.dot('div[{i} + 1]','t2[{i}]', i=ndims)};
  % endif

  Phi[0] = div[0] - gmo*invcsq*(div[0]*k - div_v + div[${nvars - 1}]);

  % for i in range(ndims - 1):
  Phi[${i + 1}] = invrho*(div_t${i+1} - div[0]*ut${i+1});
  % endfor

  Phi[${nvars - 2}] = ${invsq2}*invc*invrho*(gmo*(div[0]*k - div_v + div[${nvars - 1}]) + c*(div_norm - div[0]*un));
  Phi[${nvars - 1}] = ${invsq2}*invc*invrho*(gmo*(div[0]*k - div_v + div[${nvars - 1}]) - c*(div_norm - div[0]*un));
</%pyfr:macro>

<%pyfr:macro name='WUinv_dot_Phi-normal' params='Phi, div, u, p, v'>
  fpdtype_t rho = u[0];
  fpdtype_t invrho = 1.0/rho;
  fpdtype_t c = sqrt(${c['gamma']}*p*invrho);
  fpdtype_t invc = 1.0/c;
  fpdtype_t gmo = ${c['gamma'] - 1.0};
  fpdtype_t k = 0.5*(${pyfr.dot('v[{i}]', i=ndims)});

  fpdtype_t un = ${pyfr.dot('norm_nl[{i}]','v[{i}]', i=ndims)};
  fpdtype_t ut1 = ${pyfr.dot('t1[{i}]','v[{i}]', i=ndims)};
  % if ndims == 3:
  fpdtype_t ut2 = ${pyfr.dot('t2[{i}]','v[{i}]', i=ndims)};
  % endif

  div[0] = Phi[0] + ${invsq2}*rho*invc*(Phi[${nvars - 2}] + Phi[${nvars - 1}]);

  % for i in range(ndims):
  div[${i + 1}] =  Phi[0]*v[${i}] + rho*(${'+'.join([f'Phi[{j+1}]*t{j+1}[{i}]' for j in range(ndims-1)])}) + ${invsq2}*rho*invc*(Phi[${nvars-2}]*(c*norm_nl[${i}] + v[${i}]) - Phi[${nvars - 1}]*(c*norm_nl[${i}] - v[${i}]));
  % endfor

  div[${nvars - 1}] = k*Phi[0] +
                     rho*(${'+'.join([f'Phi[{j+1}]*ut{j+1}' for j in range(ndims-1)])}) +
                     ${invsq2}*rho*(
                                    (Phi[${nvars - 2}] + Phi[${nvars - 1}])*(c/gmo + k*invc) +
                                    (Phi[${nvars - 2}] - Phi[${nvars - 1}])*un
                                   );

</%pyfr:macro>


<%pyfr:macro name='WUinv_dot_Phi-noop' params='Phi, div, u, p, v'>
% for var in range(nvars):
div[${var}] = Phi[${var}];
% endfor
</%pyfr:macro>
<%pyfr:macro name='WU_dot_div-noop' params='div, Phi, u, p, v'>
% for var in range(nvars):
Phi[${var}] = div[${var}];
% endfor
</%pyfr:macro>

<%pyfr:kernel name='bccflux_nscbc' ndim='1'
              u_upts='in view fpdtype_t[${str(nupts)}][${str(nvars)}]'
              u_fpts='inout view fpdtype_t[${str(nfpts)}][${str(nvars)}]'
              gradu_upts='in view fpdtype_t[${str(ndims*nupts)}][${str(nvars)}]'
              smats_upts='in fpdtype_t[${str(nupts)}][${str(ndims*ndims)}]'
              jacs_ffpts='in fpdtype_t[${str(nfacefpts)}]'>

## Step 1: Compute transformed flux at solution points
fpdtype_t tf_upts[${nupts}][${ndims}][${nvars}] = {{{0}}};
% for upt in range(nupts):
{
  fpdtype_t u[${nvars}];
  fpdtype_t gradu[${ndims}][${nvars}];
  % for var in range(nvars):
    u[${var}] = u_upts[${upt}][${var}];
    % for dim in range(ndims):
    gradu[${dim}][${var}] = gradu_upts[${nuidx(dim,upt)}][${var}];
    % endfor
  % endfor
  fpdtype_t f[${ndims}][${nvars}];
  fpdtype_t p, v[${ndims}];
  ${pyfr.expand('inviscid_flux', 'u', 'f', 'p', 'v')};
  ${pyfr.expand('viscous_flux_add', 'u', 'gradu', 'f')};
  % for var in range(nvars):
    % for comp in range(ndims):
      % for phys in range(ndims):
        tf_upts[${upt}][${comp}][${var}] += smats_upts[${upt}][${nidx(comp,phys)}]*f[${phys}][${var}];
      % endfor
    % endfor
  % endfor
}
% endfor

## Step 2: Compute transformed, discontinuous, normal flux at all flux points
fpdtype_t tnf_D[${nfpts}][${nvars}] = {{0}};
% for fpt_idx in fptidx:
{
  % for upt in range(nupts):
  % for var in range(nvars):
  % for comp in range(ndims):
    % if abs(m2[fpt_idx,comp,upt]) > 0.0:
    tnf_D[${fpt_idx}][${var}] += tf_upts[${upt}][${comp}][${var}]*${m2[fpt_idx,comp,upt]};
    % endif
  % endfor
  % endfor
  % endfor
}
% endfor

## Step 3: Compute difference in divergence terms
fpdtype_t delta_div[${nfacefpts}][${nvars}];
% for f, fpt_idx in enumerate(facefpts):
{
  fpdtype_t jac = jacs_ffpts[${f}];

  ## Interpolate smats from solution points to flux point using m0
  fpdtype_t smats_fpt[${ndims}][${ndims}] = {{0}};
  % for comp in range(ndims):
    % for phys in range(ndims):
      % for upt in range(nupts):
        % if abs(m0[f,upt]) > 0.0:
  smats_fpt[${comp}][${phys}] += ${m0[f,upt]} * smats_upts[${upt}][${nidx(comp,phys)}];
        % endif
      % endfor
    % endfor
  % endfor

  ## Compute physical normal: S^T @ norm_ref
  fpdtype_t norm_nl[${ndims}];
  % for phys in range(ndims):
  norm_nl[${phys}] = 0.0;
    % for comp in range(ndims):
      % if abs(norm_ref[f,comp]) > 0.0:
  norm_nl[${phys}] += smats_fpt[${comp}][${phys}] * ${norm_ref[f,comp]};
      % endif
    % endfor
  % endfor

  ## Normalize to get unit normal
  fpdtype_t inv_mag = 1.0/sqrt(${pyfr.dot('norm_nl[{i}]', i=ndims)});
  % for phys in range(ndims):
  norm_nl[${phys}] *= inv_mag;
  % endfor

  ## Construct physical face normal CS
  % if ndims == 2:
    fpdtype_t t1[2] = {-norm_nl[1], norm_nl[0]};
  % elif ndims == 3:
    fpdtype_t sign = copysign(1.0, norm_nl[2]);
    fpdtype_t a = -1.0/(sign + norm_nl[2]);
    fpdtype_t b = norm_nl[0]*norm_nl[1]*a;
    fpdtype_t t1[3] = {1.0 + sign*norm_nl[0]*norm_nl[0]*a, sign*b, -sign*norm_nl[0]};
    fpdtype_t t2[3] = {b, sign + norm_nl[1]*norm_nl[1]*a, -norm_nl[1]};
  % endif

  ## Load state at flux point and compute primitives for characteristic decomposition
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

  ## Compute divergence in reference space using m12
  fpdtype_t div[${nvars}];
  % for var in range(nvars):
  div[${var}] = 0.0;
    % for dim in range(ndims):
      % for upt in range(nupts):
        % if abs(m12[f,dim,upt]) > 0.0:
  div[${var}] += tf_upts[${upt}][${dim}][${var}] * ${m12[f,dim,upt]};
        % endif
      % endfor
    % endfor
  % endfor

  ## Project divergence onto characteristics using physical normal
  fpdtype_t Phi[${nvars}];
  ${pyfr.expand(f'WU_dot_div-{decomp_type}','div','Phi','ul','p','v')}

  ## Apply BC to get modified wave amplitudes Phi*
  ${pyfr.expand('compute_wave_amp', 'ul', 'p', 'v', 'Phi', 'jac')};

  ## Project back to conservative divergence
  fpdtype_t div_star[${nvars}];
  ${pyfr.expand(f'WUinv_dot_Phi-{decomp_type}','Phi','div_star','ul','p','v')};

  ## Compute divergence difference
  % for var in range(nvars):
  delta_div[${f}][${var}] = div_star[${var}] - div[${var}];
  % endfor
}
% endfor

## PASS 2
% for f,fpt_idx in enumerate(facefpts):
{
  ## Compute R = GB_inv * GI * (F^\perp - F^D)
  % for var in range(nvars):
  {
    ## Compute R = GB_inv * GI * \Delta f_interior
    fpdtype_t R = 0;
    % for j, intfpt_idx in enumerate(intfpts):
      % if abs(GB_inv_GI[f,j]) > 0.0:
      R += ${GB_inv_GI[f,j]}*(u_fpts[${intfpt_idx}][${var}] - tnf_D[${intfpt_idx}][${var}]);
      % endif
    % endfor
    ## Compute A = GB_inv * d \nabla f
    fpdtype_t A = 0.0;
    % for j,fpt_jdx in enumerate(facefpts):
      % if abs(GB_inv[f,j]) > 0.0:
      A += ${GB_inv[f,j]} * delta_div[${j}][${var}];
      % endif
    % endfor

    ## Store the common normal flux in u_fpts (this is the output)
    u_fpts[${fpt_idx}][${var}] = tnf_D[${fpt_idx}][${var}] + A - R;
  }
  % endfor
}
% endfor

</%pyfr:kernel>
