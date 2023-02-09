<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='localerrest' ndim='2'
              err='in fpdtype_t[${str(nvars)}]'
              errprev='inout fpdtype_t[${str(nvars)}]'
              dtau_upts='inout fpdtype_t[${str(nvars)}]'>
    fpdtype_t ferr, gerr, ufac, vfac;

    ferr = fabs(${1/atol}*err[0]);
    gerr = errprev[0];
    ufac = ${pyfr.polyfit(lambda x: x**-expa, 1e-6, 10, 8, 'ferr')}
         * ${pyfr.polyfit(lambda x: x**expb, 1e-6, 10, 8, 'gerr')};
    vfac = min(${maxf}, max(${minf}, ${saff}*ufac));

    // Compute the size of the next step
    dtau_upts[0] = min(max(vfac*dtau_upts[0], ${dtau_min_p}), ${dtau_max_p});
    errprev[0] = ferr;

% for i in range(1,nvars):
    ferr = fabs(${1/atol}*err[${i}]);
    gerr = errprev[${i}];
    ufac = ${pyfr.polyfit(lambda x: x**-expa, 1e-6, 10, 8, 'ferr')}
         * ${pyfr.polyfit(lambda x: x**expb, 1e-6, 10, 8, 'gerr')};
    vfac = min(${maxf}, max(${minf}, ${saff}*ufac));

    // Compute the size of the next step
    dtau_upts[${i}] = min(max(vfac*dtau_upts[${i}], ${dtau_min}), ${dtau_max});
    errprev[${i}] = ferr;
% endfor
</%pyfr:kernel>
