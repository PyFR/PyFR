# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='negdivconf' ndim='2'
              t='scalar fpdtype_t'
              tdivtconf='inout fpdtype_t[${str(nvars)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              u='in fpdtype_t[${str(nvars)}]'
              rcpdjac='in fpdtype_t'>
% for i, ex in enumerate(srcex):
    tdivtconf[${i}] = -rcpdjac*tdivtconf[${i}] + ${ex};
% endfor

fpdtype_t turbsrc[${ndims}];

fpdtype_t ploc_scale[${ndims}][${ndims}] = ${ploc_scale};
fpdtype_t t_scale[${ndims}] = ${t_scale};
fpdtype_t dhat[${ndims}][${N}] = ${dhat};
fpdtype_t p[${ndims}][${N}] = ${p};
fpdtype_t q[${ndims}][${N}] = ${q};
fpdtype_t ome[${N}] = ${ome};


// the modes
fpdtype_t dhatxhat[${ndims}];
fpdtype_t arg;

% for n in range(N):
    % for i in range(ndims):
        dhatxhat[${i}] = 0.0;
        turbsrc[${i}] = 0.0;
        % for j in range(ndims):
            dhatxhat[${i}] += dhat[${j}][${n}]*ploc[${j}]*ploc_scale[${j}][${i}];
        % endfor
        arg = dhatxhat[${i}] + ome[${n}]*t*t_scale[${i}];
        turbsrc[${i}] += p[${i}][${n}]*cos(arg) + q[${i}][${n}]*sin(arg);
    % endfor
% endfor


// order is important here.
turbsrc[2] = aij[3]*turbsrc[2];
turbsrc[1] = aij[1]*turbsrc[0] + aij[2]*turbsrc[1];
turbsrc[0] = aij[0]*turbsrc[0];

// source term for synthetic turbulence, only for the momentum equations
// Multiply by the density to make it dimensionally consistent.
% for i in range(ndims):
    tdivtconf[${i} + 1] += u[0]*factor[${i}]*turbsrc[${i}];
% endfor


// TODO add pressure (i.e. energy) and density fluctuations for Ma > 0.3 flows.

</%pyfr:kernel>
