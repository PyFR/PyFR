<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='rotate_into_lhs' params='dst, src, py:rot'>
    // Rotate the RHS momentum into the LHS frame: R^T (= R^{-1})
    dst[0] = src[0];
% for i in range(ndims):
    dst[${i + 1}] = ${pyfr.constdot(rot[:, i], 'src[{j} + 1]', j=ndims)};
% endfor
    dst[${nvars - 1}] = src[${nvars - 1}];
</%pyfr:macro>

<%pyfr:macro name='rotate_from_lhs' params='dst, src, py:rot'>
    // Rotate a LHS-frame state into the RHS frame: R
    dst[0] = src[0];
% for i in range(ndims):
    dst[${i + 1}] = ${pyfr.constdot(rot[i], 'src[{j} + 1]', j=ndims)};
% endfor
    dst[${nvars - 1}] = src[${nvars - 1}];
</%pyfr:macro>

<%pyfr:macro name='rotate_grad_into_lhs' params='dst, src, py:rot'>
    // Rotate the scalar gradient into the LHS frame
<% se = f'src[{{e}}][{nvars - 1}]' %>
% for d in range(ndims):
    dst[${d}][0] = ${pyfr.constdot(rot[:, d], 'src[{e}][0]', e=ndims)};
    dst[${d}][${nvars - 1}] = ${pyfr.constdot(rot[:, d], se, e=ndims)};
% endfor

    // Rotate the momentum gradient spatial index into the LHS frame
    fpdtype_t gm[${ndims}][${ndims}];
% for d, i in pyfr.ndrange(ndims, ndims):
<% se = f'src[{{e}}][{i + 1}]' %>
    gm[${d}][${i}] = ${pyfr.constdot(rot[:, d], se, e=ndims)};
% endfor

    // Rotate the momentum gradient variable index into the LHS frame
% for d, i in pyfr.ndrange(ndims, ndims):
<% gj = f'gm[{d}][{{j}}]' %>
    dst[${d}][${i + 1}] = ${pyfr.constdot(rot[:, i], gj, j=ndims)};
% endfor
</%pyfr:macro>
