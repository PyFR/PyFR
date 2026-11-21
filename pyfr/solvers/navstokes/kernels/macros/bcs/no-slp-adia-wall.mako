<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.macros.bcs.common'/>
<%include file='pyfr.solvers.euler.kernels.macros.bcs.slp-adia-wall'/>

<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur' externs='ploc, t'>
    ur[0] = ul[0];
% for i in range(ndims):
    ur[${i + 1}] = 0.0;
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}]
                     - (0.5/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:macro name='bc_ldg_grad_state' params='ul, ur, nl, gradul, gradur'>
    ${pyfr.expand('bc_common_grad_wall', 'ul', 'ur', 'nl', 'gradul',
                  'gradur')};

    // Remove the wall-normal temperature gradient
    fpdtype_t gn = ${pyfr.dot('nl[{i}]', f'gradur[{{i}}][{nvars - 1}]',
                              i=ndims)};
% for i in range(ndims):
    gradur[${i}][${nvars - 1}] -= nl[${i}]*gn;
% endfor
</%pyfr:macro>
