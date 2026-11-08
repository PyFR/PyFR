<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='transform_grad' params='grad, smats, rcpdjac'>
    fpdtype_t tmpgrad[][${nvars}] = ${pyfr.array('grad[{i}][{j}]', i=ndims, j=nvars)};

% for i, j in pyfr.ndrange(ndims, nvars):
    grad[${i}][${j}] = rcpdjac*(${' + '.join(f'smats[{k}][{i}]*tmpgrad[{k}][{j}]'
                                              for k in range(ndims))});
% endfor
</%pyfr:macro>
