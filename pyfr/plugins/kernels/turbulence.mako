<% import numpy as np %>

<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  fpdtype_t arg, clip, g;
  fpdtype_t tpos[3];
  fpdtype_t tploc[3];
  fpdtype_t delta2[3];
  fpdtype_t utilde[3] = {};

  uint32_t oldstate, newstate, rshift;

  % for i, rotc in enumerate(rot):
    tploc[${i}] = ${' + '.join(f'{r}*(ploc[{j}] - {s})' for j, (r, s) in enumerate(zip(rotc, shift)))};
  % endfor

  % for i in range(nvmx):
    tpos[0] = ${-ls} + (t - tinit[${i}])*${avgu};

    % for j in range(2):
      oldstate = ${f'state[{i}]' if loop.index == 0 else 'newstate'};
      newstate = oldstate * 747796405UL + 2891336453UL;
      rshift = oldstate >> 28UL;
      oldstate ^= oldstate >> (4UL + rshift);
      oldstate *= 277803737UL;
      oldstate ^= oldstate >> 22UL;
      tpos[${j + 1}] = ${yzmin[j]} + ${yzdim[j]}*((fpdtype_t)(oldstate >> 8UL) * 5.9604644775390625e-8);
    % endfor

    arg = 0.0;
    % for j in range(3):
      delta2[${j}] = (tpos[${j}] - tploc[${j}])*(tpos[${j}] - tploc[${j}]);
      arg += delta2[${j}];
    % endfor

    g = (delta2[0] < ${ls**2} && delta2[1] < ${ls**2} && delta2[2] < ${ls**2} && tpos[0] <= ${ls} && state[${i}] > 0) ? ${pyfr.polyfit(lambda x: np.exp(fac1*x), 0, 3*ls**2, 8, 'arg')} : 0.0;
       
    % for j in range(3): 
      utilde[${j}] += (oldstate & ${32 << j}) ? -g : g;
    % endfor
  % endfor

  clip = (tploc[0] < ${ls} && tploc[0] > ${-ls}) ? ${pyfr.polyfit(lambda x: (fac3*avgu/ls)*np.exp(-0.5*np.pi*x**2/ls**2), -ls, ls, 8, 'tploc[0]')} : 0.0;
  
  % if not ac:
    src[0] += ${fac2}*utilde[0]*clip;
    src[${nvars - 1}] += ${0.5*fac3}*u[0]*(${pyfr.dot('utilde[{i}]', i=3)})*clip;
  % endif

  % for i in range(3):
    src[${i + 1}] += utilde[${i}]*clip${'*u[0]' if not ac else ''};
  % endfor
</%pyfr:macro>
