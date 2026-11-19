<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='bc_common_grad_zero' params='ul, ur, nl, gradul, gradur'>
% for i, j in pyfr.ndrange(ndims, nvars):
    gradur[${i}][${j}] = 0;
% endfor
</%pyfr:macro>

<%pyfr:macro name='bc_common_grad_copy' params='ul, ur, nl, gradul, gradur'>
% for i, j in pyfr.ndrange(ndims, nvars):
    gradur[${i}][${j}] = gradul[${i}][${j}];
% endfor
</%pyfr:macro>

<%pyfr:macro name='bc_common_grad_wall' params='ul, ur, nl, gradul, gradur'
             externs='gcorr'>
    fpdtype_t rcprho = 1.0/ur[0];

    // Correct the gradient as an interface with the mirrored state would
% for i, j in pyfr.ndrange(ndims, nvars):
    gradur[${i}][${j}] = gradul[${i}][${j}]
                       + gcorr*nl[${i}]*(ur[${j}] - ul[${j}]);
% endfor

    // Hold the density fixed so that no mass crosses the wall
% for i in range(ndims):
% for j in range(1, nvars):
    gradur[${i}][${j}] -= rcprho*ur[${j}]*gradur[${i}][0];
% endfor
    gradur[${i}][0] = 0;
% endfor
</%pyfr:macro>
