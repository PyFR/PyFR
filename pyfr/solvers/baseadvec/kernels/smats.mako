# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

% if ndims == 2:
<%pyfr:macro name='calc_smats_detj', params='V, x, s, d'>
    s[0][0] =  ${jac_exprs[1][1]};
    s[0][1] = -${jac_exprs[1][0]};
    s[1][0] = -${jac_exprs[0][1]};
    s[1][1] =  ${jac_exprs[0][0]};
    d = s[0][0]*s[1][1] - s[0][1]*s[1][0];
    (void) d;
</%pyfr:macro>
% else:
<%pyfr:macro name='calc_smats_detj', params='V, x, s, d'>
    fpdtype_t j[3][3];
% for i, j in pyfr.ndrange(3, 3):
    j[${i}][${j}] = ${jac_exprs[i][j]};
% endfor
% for i, (j, k) in enumerate([(1, 2), (2, 0), (0, 1)]):
    s[${i}][0] = j[${j}][1]*j[${k}][2] - j[${j}][2]*j[${k}][1];
    s[${i}][1] = j[${j}][2]*j[${k}][0] - j[${j}][0]*j[${k}][2];
    s[${i}][2] = j[${j}][0]*j[${k}][1] - j[${j}][1]*j[${k}][0];
% endfor
    d = j[0][0]*s[0][0] + j[0][1]*s[0][1] + j[0][2]*s[0][2];
    (void) d;
</%pyfr:macro>
% endif
