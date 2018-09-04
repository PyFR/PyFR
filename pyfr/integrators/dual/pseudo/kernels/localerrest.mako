# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='localerrest' ndim='2'
              err='inout fpdtype_t[${str(nvars)}]'
              errprev='inout fpdtype_t[${str(nvars)}]'
              dtau_upts ='inout fpdtype_t[${str(nvars)}]'>
    fpdtype_t fac[${nvars}];

% for i in range(nvars):
    err[${i}] = fabs(err[${i}]/${atol});

    fac[${i}] = pow(err[${i}], ${-expa}) * pow(errprev[${i}], ${expb});
    fac[${i}] = min(${maxf}, max(${minf}, ${saff}*fac[${i}]));

    // Compute the size of the next step
    dtau_upts[${i}] = min(max(fac[${i}]*dtau_upts[${i}], ${dtau_min}), ${dtau_max});
    errprev[${i}] = err[${i}];
% endfor
</%pyfr:kernel>
