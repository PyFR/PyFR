<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
${pyfr.eos_check('rusanov', fluid, 'cpg')}\
<%include file='pyfr.solvers.euler.kernels.flux'/>

<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    // Compute the left and right primitive state
    ${fluid.decl('ul', 'v, p', suffix='l')}
    ${fluid.decl('ur', 'v, p', suffix='r')}

    // Compute the left and right fluxes
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    ${pyfr.expand('inviscid_flux', 'ul', 'pl', 'vl', 'fl')};
    ${pyfr.expand('inviscid_flux', 'ur', 'pr', 'vr', 'fr')};

    // Sum the left and right velocities and take the normal
    fpdtype_t nv = ${pyfr.dot('n[{i}]', 'vl[{i}] + vr[{i}]', i=ndims)};

    // Estimate the maximum wave speed / 2
    fpdtype_t a = sqrt(${0.25*c['gamma']}*(pl + pr)/(ul[0] + ur[0]))
                + 0.25*fabs(nv);

    // Output
% for i in range(nvars):
    nf[${i}] = 0.5*(${' + '.join(f'n[{j}]*(fl[{j}][{i}] + fr[{j}][{i}])'
                                 for j in range(ndims))})
             + a*(ul[${i}] - ur[${i}]);
% endfor
</%pyfr:macro>
