<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>
<% sq2 = 2**0.5 %>
<% invsq2 = 2**-0.5 %>

<%pyfr:macro name='compute_wave_amp' params='u, p, v, Phi, jac' externs='ploc, t'>
  fpdtype_t rho = u[0];
  fpdtype_t invrho = 1.0/rho;
  fpdtype_t c = sqrt(${c['gamma']}*p*invrho);
  fpdtype_t invc = 1.0/c;

  fpdtype_t un = ${pyfr.dot('norm_nl[{i}]','v[{i}]', i=ndims)};
  fpdtype_t ut1 = ${pyfr.dot('t1[{i}]','v[{i}]', i=ndims)};

  ## Adjust target density to account for acoustic forcing contribution (isentropic relation)
  ## Incoming acoustic wave creates density perturbation: rho'_incoming = rho * u_a / c
  fpdtype_t rho_target_with_forcing = ${c['rho']} + rho*${c['u_a']}*invc;
  fpdtype_t unR = jac*${c['K_ac']}*((${c['un']} + ${c['u_a']} + ${c['u_v']}) - un);

  ## Direct injection terms (Daviller Eq. 22, Part I)
  ## Factor of 2 for acoustic, factor of 1 for vortical
  ## Need jac scaling since we're prescribing du/dt and need |J|*du/dt
  unR += jac*(2.0*${c['du_a_dt']} + ${c['du_v_dt']});

  fpdtype_t ut1R = jac*${c['K_ut']}*(0.0 - ut1);
  fpdtype_t rhoR = jac*${c['K_ac']}*(rho_target_with_forcing - rho);

  ## Need to solve for all but outgoing wave
  Phi[0] = 0.0;
  Phi[1] = -ut1R;
% if ndims == 3:
  fpdtype_t ut2 = ${pyfr.dot('t2[{i}]','v[{i}]', i=ndims)};
  fpdtype_t ut2R = jac*${c['K_ut']}*(0.0 - ut2);
  Phi[2] = -ut2R;
% endif
  Phi[${nvars - 2}] =  ${-invsq2}*(unR + c*invrho*rhoR);

</%pyfr:macro>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t'>
  % for i in range(nvars):
    ur[${i}] = ul[${i}];
  % endfor
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>