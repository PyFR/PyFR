<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>
<% sq2 = 2**0.5 %>
<% invsq2 = 2**-0.5 %>

<%pyfr:macro name='compute_wave_amp' params='u, p, v, Phi, jac' externs='ploc, t'>
  fpdtype_t rho = u[0];
  fpdtype_t invrho = 1.0/rho;
  fpdtype_t c = sqrt(${c['gamma']}*p*invrho);
  fpdtype_t invc = 1.0/c;

  fpdtype_t nx = norm_nl[0];
  fpdtype_t ny = norm_nl[1];

  fpdtype_t rhoR = jac*${c['K_rho']}*(${c['rho']} - rho);
  fpdtype_t uR = jac*${c['K_u']}*(${c['u']} - v[0]);
  fpdtype_t vR = jac*${c['K_v']}*(${c['v']} - v[1]);

  ## Need to solve for all but outgoing wave
% if ndims == 2:

  fpdtype_t veloR = nx*uR + ny*vR;
  Phi[0] = rho*invc*veloR - rhoR;
  Phi[1] = uR*ny - vR*nx;
  Phi[2] = ${-sq2}*veloR;

% elif ndims == 3:

  fpdtype_t nz = norm_nl[2];
  fpdtype_t wR = jac*${c['K_w']}*(${c['w']} - v[2]);

  fpdtype_t veloR = nx*uR + ny*vR + nz*wR;
  Phi[0] =  invc*nx*rho*veloR - (nx*rhoR + nz*vR - ny*wR);
  Phi[1] =  invc*ny*rho*veloR - (ny*rhoR - nz*uR + nx*wR);
  Phi[2] =  invc*nz*rho*veloR - (nz*rhoR + ny*uR - nx*vR);
  Phi[3] =  ${-sq2}*veloR;

% endif
</%pyfr:macro>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t'>
  % for i in range(nvars):
    ur[${i}] = ul[${i}];
  % endfor
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>