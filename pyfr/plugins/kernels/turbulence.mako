<%namespace module='numpy' name='np'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  fpdtype_t tbc = 5.9604644775390625e-8;
  fpdtype_t arg;
  fpdtype_t xloc2;
  fpdtype_t clip;
  fpdtype_t g;
  fpdtype_t tpos[3];
  fpdtype_t tploc[3];
  fpdtype_t delta2[3];
  fpdtype_t yzmin[2] = {${ymin}, ${zmin}};
  fpdtype_t yzmax[2] = {${ymax}, ${zmax}};
  fpdtype_t utilde[3] = {};

  unsigned int oldstate;
  unsigned int newstate;
  unsigned int rshift;
  
  % for i, r in enumerate(rot):
    tploc[${i}] = ${' + '.join(f'{r[j]}*(ploc[{j}] - {shift[j]})' for j in range(3))};
  % endfor

  % for i in range(nvmax):
    tpos[0] = ${-ls} + (t-tinit[${i}][0])*${ubar};

    % for j in range(2):
      % if loop.index == 0:
        oldstate = state[${i}][0];
      % else:
        oldstate = newstate;
      % endif
      newstate = (oldstate * 747796405UL) + 2891336453UL;
      rshift = oldstate >> (28UL);
      oldstate ^= oldstate >> (4UL + rshift);
      oldstate *= 277803737UL;
      oldstate ^= oldstate >> 22UL;
      tpos[${j+1}] = yzmin[${j}] + (yzmax[${j}]-yzmin[${j}])*((fpdtype_t)(oldstate >> 8UL) * tbc);
    % endfor

    arg = 0.0;
    % for j in range(3):
      delta2[${j}] = (tpos[${j}] - tploc[${j}])*(tpos[${j}] - tploc[${j}]);
      arg += ${fac1}*delta2[${j}];
    % endfor

    g = (delta2[0] < ${ls**2} && delta2[1] < ${ls**2} && delta2[2] < ${ls**2} && tpos[0] <= ${ls}) ? ${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'arg')} : 0.0;

    g = (state[${i}][0] > 0) ? g : 0.0;
       
    % for j in range(3): 
      utilde[${j}] += ((oldstate & ${32 << j}) ? -1 : 1)*g;
    % endfor
  % endfor

  xloc2 = tploc[0]*tploc[0]*${-0.5*math.pi/(ls**2)};
     
  clip = (tploc[0] < ${ls} && tploc[0] > ${-ls}) ? ${ubar/ls}*${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'xloc2')} : 0.0;
  
  src[0] += ${fac2}*${fac3}*utilde[0]*clip;
  % for i in range(3):
    src[${i + 1}] += u[0]*${fac3}*utilde[${i}]*clip;
  % endfor
  fpdtype_t udotu_fluct = ${pyfr.dot('utilde[{i}]', i=3)};     
  src[${nvars - 1}] += 0.5*u[0]*udotu_fluct*clip*${fac3**2};
</%pyfr:macro>
