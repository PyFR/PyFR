# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:function name='inviscid_flux'
                params='const fpdtype_t s[${str(nvars)}],
                        fpdtype_t f[${str(ndims)}][${str(nvars)}],
                        fpdtype_t pout[1], fpdtype_t vout[${str(ndims)}]'>
    fpdtype_t rho = s[0], invrho = 1.0/s[0], E = s[${nvars - 1}];

    // Compute the velocities
    fpdtype_t rhov[] = ${pyfr.array('s[{i}]', i=(1, ndims + 1))};
    fpdtype_t v[] = ${pyfr.array('invrho*rhov[{i}]', i=ndims)};

    // Compute the pressure
    fpdtype_t p = ${c['gamma'] - 1}*(E - 0.5*invrho*${pyfr.dot('rhov[{i}]',
                                                               i=ndims)});

    // Density and energy fluxes
% for i in xrange(ndims):
    f[${i}][0] = rhov[${i}];
    f[${i}][${nvars - 1}] = (E + p)*v[${i}];
% endfor

    // Momentum fluxes
% for i, j in pyfr.ndrange(ndims, ndims):
    f[${i}][${j + 1}] = rhov[${i}]*v[${j}]${' + p' if i == j else ''};
% endfor

    if (pout != NULL)
        pout[0] = p;

    if (vout != NULL)
    {
% for i in xrange(ndims):
        vout[${i}] = v[${i}];
% endfor
    }
</%pyfr:function>
