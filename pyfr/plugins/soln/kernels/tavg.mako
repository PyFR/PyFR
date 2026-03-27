<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='${eos_mod}'/>

typedef ${'fpdtype_t' if use_kahan else 'double'} acctype_t;

<%pyfr:kernel name='tavg' ndim='2'
              u='in ${"view " if use_views else ""}fpdtype_t[${str(nvars)}]'
              gradu='in ${"view " if use_views else ""}fpdtype_t[${str(ndims)}][${str(nvars)}]'
              acc='inout acctype_t[${str(nexprs)}]'
              acc_comp='inout fpdtype_t[${str(nexprs)}]'
              vacc='inout acctype_t[${str(nexprs)}]'
              prev='inout fpdtype_t[${str(nexprs)}]'
              wdt='scalar fpdtype_t'
              wacc='scalar fpdtype_t'
              wvar='scalar fpdtype_t'>
    fpdtype_t pri[${nvars}];
    ${pyfr.expand('con_to_pri', 'u', 'pri')};
% if has_grads:
    fpdtype_t grad_pri[${nvars}][${ndims}];
    ${pyfr.expand('grad_con_to_pri', 'u', 'gradu', 'grad_pri')};
% endif
% for i, expr in enumerate(exprs):
    {
        fpdtype_t curr = ${expr};
        fpdtype_t ppc = prev[${i}] + curr;
% if use_kahan:
        acctype_t nacc;
        <%pyfr:fp_precise>
            fpdtype_t y = wdt*ppc - acc_comp[${i}];
            nacc = acc[${i}] + y;
            acc_comp[${i}] = (nacc - acc[${i}]) - y;
        </%pyfr:fp_precise>
% else:
        acctype_t nacc = acc[${i}] + wdt*ppc;
% endif
% if has_var:
        acctype_t nvacc = vacc[${i}]
            + wdt*(prev[${i}]*prev[${i}] + curr*curr - 0.5*ppc*ppc);
        if (wvar > 0)
            nvacc += (wdt/wvar)*(nacc - wacc*ppc)*(nacc - wacc*ppc);
        vacc[${i}] = nvacc;
% endif
        acc[${i}] = nacc;
        prev[${i}] = curr;
    }
% endfor
</%pyfr:kernel>
