# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  fpdtype_t tbc = 2.3283064365386962890625e-10;
  fpdtype_t arg;  
  fpdtype_t xloc2;
  fpdtype_t sclip;
  fpdtype_t g;
  fpdtype_t pos[${ndims}];
  fpdtype_t ttploc[${ndims}];
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
    ttploc[${i}] = ${' + '.join(f'{r[j]}*(ploc[{j}] - {shift[j]})' for j in range(3))};
  % endfor

  % for i in range(nvmax):
    pos[0] = ${-ls} + (t-tinit[${i}][0])*${ubar};

    % for j in range(3):
      % if loop.index == 0:
        oldstate = state[${i}][0];
      % else:
        oldstate = newstate;
      % endif
      newstate = (oldstate * 747796405UL) + 2891336453UL;
      rshift = oldstate >> (32UL - 4UL);
      oldstate ^= oldstate >> (4UL + rshift);
      oldstate *= 277803737UL;
      oldstate ^= oldstate >> 22UL;
      % if loop.index == 0:
        pos[1] = ${ymin} + ${ymax-ymin}*((fpdtype_t)oldstate * tbc);
      % elif loop.index == 1:
        pos[2] = ${zmin} + ${zmax-zmin}*((fpdtype_t)oldstate * tbc);
      % elif loop.index == 2:
        epscomp = oldstate % 8;
      % endif
    % endfor

    arg = 0.0;
    % for j in range(ndims):
      delta2[${j}] = (pos[${j}]-ttploc[${j}])*(pos[${j}]-ttploc[${j}]);
      arg += ${-0.5/(sigma*sigma*ls*ls)}*delta2[${j}];
    % endfor

    g = (delta2[0] < ${ls*ls} && delta2[1] < ${ls*ls} && delta2[2] < ${ls*ls} && pos[0] <= ${ls}) ? ${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'arg')} : 0.0;

    g = state[${i}][0] > 0 ? g : 0.0;
       
    % for j in range(ndims): 
      utilde[${j}] += ((epscomp & 1 << ${j}) ? -1 : 1)*g;
    % endfor
  % endfor
    
  xloc2 = ttploc[0]*ttploc[0]*${-0.5*math.pi/(ls*ls)};
  
  sclip = ttploc[0] < ${ls} ? ttploc[0] > ${-ls} ? ${ubar/ls}*${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'xloc2')} : 0.0: 0.0;
  
  src[0] += ${srafac*fac}*utilde[0]*sclip;
  % for i in range(ndims):
    src[${i + 1}] += u[0]*${fac}*utilde[${i}]*sclip;
  % endfor
  fpdtype_t udotu_fluct = ${pyfr.dot('utilde[{i}]', i=(0, ndims))};     
  src[${nvars - 1}] += 0.5*u[0]*udotu_fluct*sclip*${fac*fac};
</%pyfr:macro>
