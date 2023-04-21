# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='turbulence' params='t, u, ploc, src'>
  fpdtype_t xin = 0.0;
  fpdtype_t ls = ${ls};
  fpdtype_t ls2 = ${ls*ls};
  fpdtype_t invls2 = ${1.0/(ls*ls)};
  fpdtype_t gc3 = ${gc*gc*gc};
  fpdtype_t rootrs = ${rootrs};
  fpdtype_t srafac = ${srafac};
  fpdtype_t invsigma2 = ${1.0/(sigma*sigma)};
  fpdtype_t invsigma3 = ${1.0/(sigma*sigma*sigma)};
  fpdtype_t ubar = ${ubar};
  fpdtype_t pos[${ndims}];
  fpdtype_t ttploc[${ndims}];
  fpdtype_t eps[${ndims}];
  fpdtype_t delta2[${ndims}];
  fpdtype_t arg;
  fpdtype_t utilde[${ndims}];
  utilde[0] = 0.0;
  utilde[1] = 0.0;
  utilde[2] = 0.0;
  fpdtype_t xloc2;
  fpdtype_t sclip;
  fpdtype_t g;
  fpdtype_t xmin = -${ls};
  fpdtype_t xmax = ${ls};
  fpdtype_t ymin = ${ymin};
  fpdtype_t ymax = ${ymax};
  fpdtype_t zmin = ${zmin};
  fpdtype_t zmax = ${zmax};
  fpdtype_t fac = ${-0.5/(sigma*sigma*ls*ls)};
  fpdtype_t fac2 = ${(gc*gc*gc)/(sigma*sigma*sigma)};

  fpdtype_t tbc = 2.3283064365386962890625e-10;
  
  unsigned int oldstate;
  unsigned int newstate;
  unsigned int rshift;
  unsigned int b22 = 22;
  unsigned int b32 = 32;
  unsigned int opbits = 4;

  int epscomp;
  
  % for i, r in enumerate(rot):
    ttploc[${i}] = ${' + '.join(f'{r[j]}*(ploc[{j}] - {shift[j]})' for j in range(3))};
  % endfor

  % for i in range(nvmax):
    pos[0] = xmin + (t-tinit[${i}][0])*ubar;
    
    oldstate = state[${i}][0];
    newstate = (oldstate * 747796405UL) + 2891336453UL;
    rshift = oldstate >> (b32 - opbits);
    oldstate ^= oldstate >> (opbits + rshift);
    oldstate *= 277803737UL;
    oldstate ^= oldstate >> b22;
    pos[1] = ymin + (ymax-ymin)*((fpdtype_t)oldstate * tbc);
    //pos[1] = ymin + (ymax-ymin)*ldexp((fpdtype_t)oldstate, -32);
    
    oldstate = newstate;
    newstate = (oldstate * 747796405UL) + 2891336453UL;
    rshift = oldstate >> (b32 - opbits);
    oldstate ^= oldstate >> (opbits + rshift);
    oldstate *= 277803737UL;
    oldstate ^= oldstate >> b22;
    pos[2] = zmin + (zmax-zmin)*((fpdtype_t)oldstate * tbc);
    
    oldstate = newstate;
    newstate = (oldstate * 747796405UL) + 2891336453UL;
    rshift = oldstate >> (b32 - opbits);
    oldstate ^= oldstate >> (opbits + rshift);
    oldstate *= 277803737UL;
    oldstate ^= oldstate >> b22;
    epscomp = oldstate % 8;

    arg = 0.0;
    % for j in range(ndims):
      delta2[${j}] = (pos[${j}]-ttploc[${j}])*(pos[${j}]-ttploc[${j}]);
      arg += fac*delta2[${j}];
    % endfor

    g = delta2[0] < ls2 ? delta2[1] < ls2 ? delta2[2] < ls2 ? pos[0] <= xmax ? fac2*${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'arg')} : 0.0 : 0.0 : 0.0 : 0.0;
    
    eps[0] = (epscomp & 1) ? -1 : 1;
    eps[1] = (epscomp & 2) ? -1 : 1;
    eps[2] = (epscomp & 4) ? -1 : 1;
       
    % for j in range(ndims): 
      utilde[${j}] += eps[${j}]*g;
    % endfor
  % endfor
  
  % for i in range(ndims): 
    utilde[${i}] *= rootrs;
  % endfor
  
  xloc2 = -0.5*3.141592654*ttploc[0]*ttploc[0]*invls2;
  
  sclip = ttploc[0] < xmax ? ttploc[0] > xmin ? (ubar/ls)*${pyfr.polyfit(lambda x: 2.718281828459045**x, 0, 1, 8, 'xloc2')} : 0.0: 0.0;
  
  src[0] += srafac*utilde[0]*sclip;
  % for i in range(ndims):
    src[${i+1}] += u[0]*utilde[${i}]*sclip;
  % endfor
  fpdtype_t udotu_fluct = ${pyfr.dot('utilde[{i}]', i=(0, ndims))};     
  src[${nvars-1}] += 0.5*u[0]*udotu_fluct*sclip;
  
</%pyfr:macro>
