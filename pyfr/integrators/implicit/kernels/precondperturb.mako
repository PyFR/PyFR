<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='precondperturb' ndim='1'
              u='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              up='inout fpdtype_t[${str(nupts)}][${str(nvars)}]'
              colours='in ixdtype_t'
              pcolour='scalar ixdtype_t'
              pcidx='scalar ixdtype_t'
              colour='scalar ixdtype_t'
              cidx='scalar ixdtype_t'
              eps='scalar fpdtype_t'>
    // Per-variable scaling for FD perturbation
    const fpdtype_t _escale[] = ${pyfr.carray(eps_scales)};

    // Reset previous perturbation
    if (pcolour >= 0 && colours == pcolour)
        up[pcidx / ${nvars}][pcidx % ${nvars}] = u[pcidx / ${nvars}][pcidx % ${nvars}];

    // Perturb current position with scaled eps
    if (colours == colour)
    {
        int k = cidx % ${nvars};
        up[cidx / ${nvars}][k] = u[cidx / ${nvars}][k] + eps*_escale[k];
    }
</%pyfr:kernel>
