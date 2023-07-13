# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  fpdtype_t tbc = 5.9604644775390625e-8;
  fpdtype_t arg;  
  fpdtype_t xloc2;
  fpdtype_t sclip;
  fpdtype_t g;
  fpdtype_t pos[${ndims}];
  fpdtype_t tploc[${ndims}];
  fpdtype_t eps[${ndims}];
  fpdtype_t delta2[${ndims}];
  fpdtype_t utilde[${ndims}];

  utilde[0] = 0.0;
  utilde[1] = 0.0;
  utilde[2] = 0.0;

  unsigned int oldstate;
  unsigned int newstate;
  unsigned int rshift;

  int epscomp;
  
  % for i, r in enumerate(rot):
    tploc[${i}] = ${' + '.join(f'{r[j]}*(ploc[{j}] - {shift[j]})' for j in range(3))};
  % endfor

  % for i in range(nvmax):
    pos[0] = ${-ls} + (t-tinit[${i}][0])*${ubar};

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
      % if loop.index == 0:
        pos[1] = ${ymin} + ${ymax-ymin}*((fpdtype_t)(oldstate >> 8UL) * tbc);
      % else:
        pos[2] = ${zmin} + ${zmax-zmin}*((fpdtype_t)(oldstate >> 8UL) * tbc);
      % endif
    % endfor
    epscomp = oldstate;

    arg = 0.0;
    % for j in range(ndims):
      delta2[${j}] = (pos[${j}]-tploc[${j}])*(pos[${j}]-tploc[${j}]);
      arg += ${-0.5/(sigma*sigma*ls*ls)}*delta2[${j}];
    % endfor

    g = (delta2[0] < ${ls*ls} && delta2[1] < ${ls*ls} && delta2[2] < ${ls*ls} && pos[0] <= ${ls}) ? ${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'arg')} : 0.0;

    g = state[${i}][0] > 0 ? g : 0.0;
       
    % for j in range(ndims): 
      utilde[${j}] += ((epscomp & (32UL << ${j}UL)) ? -1 : 1)*g;
    % endfor
  % endfor
    
  xloc2 = tploc[0]*tploc[0]*${-0.5*math.pi/(ls*ls)};
  
  sclip = tploc[0] < ${ls} ? tploc[0] > ${-ls} ? ${ubar/ls}*${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'xloc2')} : 0.0: 0.0;
  
  src[0] += ${srafac*fac}*utilde[0]*sclip;
  % for i in range(ndims):
    src[${i + 1}] += u[0]*${fac}*utilde[${i}]*sclip;
  % endfor
  fpdtype_t udotu_fluct = ${pyfr.dot('utilde[{i}]', i=(0, ndims))};     
  src[${nvars - 1}] += 0.5*u[0]*udotu_fluct*sclip*${fac*fac};
</%pyfr:macro>
