# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<%pyfr:kernel name='tflux' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              f='out fpdtype_t[${str(ndims)}][${str(nvars)}]'>
    // Compute the flux
    fpdtype_t ftemp[${ndims}][${nvars}];
    fpdtype_t p, v[${ndims}];
    ${pyfr.expand('inviscid_flux', 'u', 'ftemp', 'p', 'v')};

    // Transform the fluxes
% for i, j in pyfr.ndrange(ndims, nvars):
    f[${i}][${j}] = ${' + '.join('smats[{0}][{1}]*ftemp[{1}][{2}]'
                                 .format(i, k, j)
                                 for k in range(ndims))};
% endfor
</%pyfr:kernel>
