# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='entropy' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              s='out fpdtype_t'>
    fpdtype_t invrho = 1.0/u[0], E = u[${nvars - 1}];

    // Compute the pressure
    fpdtype_t p = ${c['gamma'] - 1}*(E - 0.5*invrho*${pyfr.dot('u[{i}]', i=(1, ndims + 1))});

    // Compute Entropy
    s = p*pow(invrho, ${c['gamma']});
</%pyfr:kernel>
