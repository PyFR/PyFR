<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>
<% sq2 = 2**0.5 %>
<% invsq2 = 2**-0.5 %>

<%pyfr:macro name='compute_wave_amp' params='u, p, v, Phi, jac' externs='ploc, t'>
  fpdtype_t invrho = 1.0/u[0];
  fpdtype_t c = sqrt(${c['gamma']}*p*invrho);
  fpdtype_t invc = 1.0/c;

  fpdtype_t pR = jac*${c['K_p']}*(${c['p']} - p);

  Phi[${nvars-1}] = ${-sq2}*invc*invrho*pR;

</%pyfr:macro>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t'>
  % for i in range(nvars):
    ur[${i}] = ul[${i}];
  % endfor
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>