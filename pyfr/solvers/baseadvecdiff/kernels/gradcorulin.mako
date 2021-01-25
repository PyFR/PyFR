# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.baseadvec.kernels.smats'/>

<%pyfr:kernel name='gradcorulin' ndim='2'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              verts='in broadcast-col fpdtype_t[${str(nverts)}][${str(ndims)}]'
              upts='in broadcast-row fpdtype_t[${str(ndims)}]'>
    // Compute the S matrices
    fpdtype_t smats[${ndims}][${ndims}], djac;
    ${pyfr.expand('calc_smats_detj', 'verts', 'upts', 'smats', 'djac')};

    fpdtype_t rcpdjac = 1 / djac;
    fpdtype_t tmpgradu[${ndims}];

% for j in range(nvars):
% for i in range(ndims):
    tmpgradu[${i}] = gradu[${i}][${j}];
% endfor
% for i in range(ndims):
    gradu[${i}][${j}] = rcpdjac*(${' + '.join(f'smats[{k}][{i}]*tmpgradu[{k}]'
                                              for k in range(ndims))});
% endfor
% endfor
</%pyfr:kernel>
