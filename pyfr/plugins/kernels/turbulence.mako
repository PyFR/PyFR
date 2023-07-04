<% import numpy as np %>

<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src' externs='tinit, state'>
  fpdtype_t arg, clip, g;
  fpdtype_t tpos[3], tploc[3], delta2[3], utilde[3] = {};

  uint32_t oldstate, newstate, rshift;

  % for i, rotc in enumerate(rot):
    tploc[${i}] = ${' + '.join(f'{r}*(ploc[{j}] - {s})' for j, (r, s) in enumerate(zip(rotc, shift)))};
  % endfor

  % for i in range(nvmx):
    tpos[0] = ${-ls} + (t - tinit[${i}])*${avgu};

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
      delta2[${j}] = (tpos[${j}] - tploc[${j}])*(tpos[${j}] - tploc[${j}]);
      arg += delta2[${j}];
    % endfor

    g = (delta2[0] < ${ls**2} && delta2[1] < ${ls**2} && delta2[2] < ${ls**2} && tpos[0] <= ${ls} && state[${i}] > 0) ? ${pyfr.polyfit(lambda x: np.exp(beta1*x), 0, 3*ls**2, 8, 'arg')} : 0.0;
       
    % for j in range(3): 
      utilde[${j}] += (oldstate & ${32 << j}) ? -g : g;
    % endfor
  % endfor

  clip = (tploc[0] < ${ls} && tploc[0] > ${-ls}) ? ${pyfr.polyfit(lambda x: (beta3*avgu/ls)*np.exp(-0.5*np.pi*x**2/ls**2), -ls, ls, 8, 'tploc[0]')} : 0.0;
  
  % if not ac:
    src[0] += ${beta2}*utilde[0]*clip;
    src[${nvars - 1}] += ${0.5*beta3}*u[0]*(${pyfr.dot('utilde[{i}]', i=3)})*clip;
  % endif

  % for i in range(3):
    src[${i + 1}] += utilde[${i}]*clip${'*u[0]' if not ac else ''};
  % endfor
</%pyfr:macro>
