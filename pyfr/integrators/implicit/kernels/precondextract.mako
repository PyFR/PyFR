<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='precondextract' ndim='1'
              f='in view fpdtype_t[${str(nupts)}][${str(nvars)}]'
              f0='in view fpdtype_t[${str(nupts)}][${str(nvars)}]'
              jac='out ${wkdtype}[${str(n)}][${str(n)}]'
              cidx='scalar ixdtype_t'
              inv_eps='scalar fpdtype_t'>
    for (int upt = 0; upt < ${nupts}; upt++)
        for (int var = 0; var < ${nvars}; var++)
        {
            int r = upt*${nvars} + var;
            jac[r][cidx] = ${pyfr.fpcast('(f[upt][var] - f0[upt][var])*inv_eps', 'fpdtype_t', wkdtype)};
        }
</%pyfr:kernel>
