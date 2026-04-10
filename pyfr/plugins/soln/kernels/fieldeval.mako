<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='${eos_mod}'/>

<%pyfr:kernel name='fieldeval' ndim='2'
              u='in ${"view " if use_views else ""}fpdtype_t[${str(nvars)}]'
              gradu='in ${"view " if use_views else ""}fpdtype_t[${str(ndims)}][${str(nvars)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              wts='in fpdtype_t'
              out='out broadcast-col reduce(${reduceop}) fpdtype_t[${str(nexprs)}]'
              t='scalar fpdtype_t'>
    fpdtype_t pri[${nvars}];
    ${pyfr.expand('con_to_pri', 'u', 'pri')};

% if has_grads:
    fpdtype_t grad_pri[${nvars}][${ndims}];
    ${pyfr.expand('grad_con_to_pri', 'u', 'gradu', 'grad_pri')};
% endif

% for j, expr in enumerate(exprs):
% if reduceop == 'sum':
    out[${j}] = wts*(${expr});
% elif has_wts:
    out[${j}] = wts > 0 ? (${expr}) : ${'-' if reduceop == 'max' else ''}${fpdtype_max};
% else:
    out[${j}] = ${expr};
% endif
% endfor
</%pyfr:kernel>
