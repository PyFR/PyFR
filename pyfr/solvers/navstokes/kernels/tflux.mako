# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<%pyfr:kernel name='tflux' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims**2)}]'
              rcpdjac='in fpdtype_t'
              f='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'>
    // Un-transform the transformed gradients (stored in f)
    fpdtype_t gradu[${ndims}][${nvars}];
% for i, j in pyfr.ndrange(ndims, nvars):
    gradu[${i}][${j}] = rcpdjac*(${' + '.join('smats[{0}]*f[{1}][{2}]'
                                              .format(k*ndims + i, k, j)
                                              for k in range(ndims))});
% endfor

    // Compute the flux (F = Fi + Fv)
    fpdtype_t ftemp[${ndims}][${nvars}];
    inviscid_flux(u, ftemp, NULL, NULL);
    viscous_flux_add(u, gradu, ftemp);

    // Transform the flux
% for i, j in pyfr.ndrange(ndims, nvars):
    f[${i}][${j}] = ${' + '.join('smats[{0}]*ftemp[{1}][{2}]'
                                 .format(i*ndims + k, k, j)
                                 for k in range(ndims))};
% endfor
</%pyfr:kernel>
