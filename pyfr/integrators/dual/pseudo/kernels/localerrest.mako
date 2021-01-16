# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='localerrest' ndim='2'
              err='in fpdtype_t[${str(nvars)}]'
              errprev='inout fpdtype_t[${str(nvars)}]'
              dtau_upts='inout fpdtype_t[${str(nvars)}]'>
    fpdtype_t ferr, ufac, vfac;

% for i in range(nvars):
    ferr = fabs(${1/atol}*err[${i}]);
    ufac = pow(ferr, ${-expa}) * pow(errprev[${i}], ${expb});
    vfac = min(${maxf}, max(${minf}, ${saff}*ufac));

    // Compute the size of the next step
    dtau_upts[${i}] = min(max(vfac*dtau_upts[${i}], ${dtau_min}), ${dtau_max});
    errprev[${i}] = ferr;
% endfor
</%pyfr:kernel>
