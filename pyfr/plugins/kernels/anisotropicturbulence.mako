<% import numpy as np %>

<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='anisotropicturbulence' params='t, u, ploc, src' externs='tinit, state'>
  fpdtype_t arg, clip, g;
  fpdtype_t tpos[3], tploc[3], delta[3], delta2[3], xi[3], utilde[3] = {}, u_fluct[3] = {};

  uint32_t oldstate, newstate, rshift;


  % for i, rotc in enumerate(rot):
    tploc[${i}] = ${' + '.join(f'{r}*(ploc[{j}] - {s})' for j, (r, s) in enumerate(zip(rotc, shift)))};
  % endfor

  % for i in range(nvmx):
    tpos[0] = ${-ls[0]} + (t - tinit[${i}])*${avgu};

    % for j in range(2):
      oldstate = ${f'state[{i}]' if loop.index == 0 else 'newstate'};
      newstate = oldstate * 747796405 + 2891336453;
      rshift = oldstate >> 28;
      oldstate ^= oldstate >> (4 + rshift);
      oldstate *= 277803737;
      oldstate ^= oldstate >> 22;
      tpos[${j + 1}] = ${yzdim[j]}*(-0.5 + (fpdtype_t)(oldstate >> 8)*${pow(2, -24)});
    % endfor

    arg = 0.0;
    % for j in range(3):
      delta [${j}] = tpos[${j}] - tploc[${j}];
      xi[${j}]     = delta[${j}]/${ls[j]};
      delta2[${j}] = delta[${j}]*delta[${j}];
      arg += delta2[${j}]*${beta1[j]};
    % endfor

    g = (delta2[0] < ${ls[0]**2} && delta2[1] < ${ls[1]**2} && delta2[2] < ${ls[2]**2} && tpos[0] <= ${ls[0]} && state[${i}] > 0) ? ${pyfr.polyfit(lambda x: np.exp(x), -100, 0.0, 8, 'arg')} : 0.0;
      
    % for j in range(3):
      % if j == 1:
        /* DEBUG: kill v */
        /* do nothing */
      % else:    
        if (fabs(xi[0]) <= ${xi_max[0][j]} && 
            fabs(xi[1]) <= ${xi_max[1][j]} && 
            fabs(xi[2]) <= ${xi_max[2][j]})
          utilde[${j}] += (oldstate & ${32 << j}) ? -g : g;
      % endif
    % endfor
  % endfor

  clip = (tploc[0] < ${ls[0]} && tploc[0] > ${-ls[0]}) ? ${pyfr.polyfit(lambda x: (avgu/ls[0])*np.exp(-0.5*np.pi*x**2/ls[0]**2), -ls[0], ls[0], 8, 'tploc[0]')} : 0.0;

  u_fluct[0] = ${beta3[0][0]}*utilde[0];
  u_fluct[1] = ${beta3[1][0]}*utilde[0] + ${beta3[1][1]}*utilde[1];
  u_fluct[2] = ${beta3[2][0]}*utilde[0] + ${beta3[2][1]}*utilde[1] + ${beta3[2][2]}*utilde[2];

  % if not ac:
    src[0] += ${beta2}*utilde[0]*clip;  
    src[${nvars - 1}] += u[0] * (u[1]/u[0]*u_fluct[0] + u[2]/u[0]*u_fluct[1] + u[3]/u[0]*u_fluct[2]) * clip;
  % endif

  % for i in range(3):
    src[${i + 1}] += u_fluct[${i}]*clip${'*u[0]' if not ac else ''};
  % endfor



</%pyfr:macro>
