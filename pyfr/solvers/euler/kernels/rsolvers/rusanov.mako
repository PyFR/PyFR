# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<%pyfr:function name='rsolve'
                params='const fpdtype_t ul[${str(nvars)}],
                        const fpdtype_t ur[${str(nvars)}],
                        const fpdtype_t n[${str(ndims)}],
                        fpdtype_t nf[${str(nvars)}]'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}];
    fpdtype_t pl, pr;

    inviscid_flux(ul, fl, &pl, vl);
    inviscid_flux(ur, fr, &pr, vr);

    // Sum the left and right velocities and take the normal
    fpdtype_t nv = ${pyfr.dot('n[{i}]', 'vl[{i}] + vr[{i}]', i=ndims)};

    // Estimate the maximum wave speed / 2
    fpdtype_t a = sqrt(${0.25*c['gamma']}*(pl + pr)/(ul[0] + ur[0]))
                + 0.25*fabs(nv);

    // Output
    for (int i = 0; i < ${nvars}; i++)
        nf[i] = 0.5*${pyfr.dot('n[{j}]', 'fl[{j}][i] + fr[{j}][i]', j=ndims)}
              + a*(ul[i] - ur[i]);
</%pyfr:function>
