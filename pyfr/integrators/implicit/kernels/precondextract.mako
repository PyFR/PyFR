<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='precondextract' ndim='1'
              f='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              f0='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              J='inout fpdtype_t[${str(bsize)}][${str(bsize)}]'
              colours='in ixdtype_t'
              colour='scalar ixdtype_t'
              cidx='scalar ixdtype_t'
              eps='scalar fpdtype_t'>
    // Per-variable scaling for FD perturbation
    const fpdtype_t _escale[] = ${pyfr.carray(eps_scales)};

    if (colours == colour)
    {
        int k = cidx % ${nvars};
        fpdtype_t inv_eps = 1 / (eps*_escale[k]);
        for (int ii = 0; ii < ${nupts}; ii++)
            for (int kk = 0; kk < ${nvars}; kk++)
                J[ii*${nvars} + kk][cidx] = (f[ii][kk] - f0[ii][kk]) * inv_eps;
    }
</%pyfr:kernel>
