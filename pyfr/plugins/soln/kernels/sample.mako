<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='${eos_mod}'/>

<%pyfr:kernel name='sample' ndim='1'
              u='in view fpdtype_t[${str(nupts)}][${str(nvars)}]'
              gradu='in view fpdtype_t[${str(ndims*nupts)}][${str(nvars)}]'
              wts='in fpdtype_t[${str(nupts)}]'
              out='out fpdtype_t[${str(nsvars)}]'>
    // Interpolate conservative solution to sample point
    fpdtype_t upt[${nvars}] = {};
% for j in range(nupts):
    {
        fpdtype_t w = wts[${j}];
% for v in range(nvars):
        upt[${v}] += w*u[${j}][${v}];
% endfor
    }
% endfor

% if has_grads:
    // Interpolate conservative gradients to sample point
    fpdtype_t gradupt[${ndims}][${nvars}] = {};
% for d in range(ndims):
% for j in range(nupts):
    {
        fpdtype_t w = wts[${j}];
% for v in range(nvars):
        gradupt[${d}][${v}] += w*gradu[${d*nupts + j}][${v}];
% endfor
    }
% endfor
% endfor
% endif

% if primitive:
    // Convert to primitive variables
    fpdtype_t pri[${nvars}];
    ${pyfr.expand('con_to_pri', 'upt', 'pri')};
% for v in range(nvars):
    out[${v}] = pri[${v}];
% endfor
% if has_grads:
    fpdtype_t grad_pri[${nvars}][${ndims}];
    ${pyfr.expand('grad_con_to_pri', 'upt', 'gradupt', 'grad_pri')};
% for v in range(nvars):
% for d in range(ndims):
    out[${nvars + v*ndims + d}] = grad_pri[${v}][${d}];
% endfor
% endfor
% endif
% else:
    // Output conservative variables directly
% for v in range(nvars):
    out[${v}] = upt[${v}];
% endfor
% if has_grads:
% for v in range(nvars):
% for d in range(ndims):
    out[${nvars + v*ndims + d}] = gradupt[${d}][${v}];
% endfor
% endfor
% endif
% endif
</%pyfr:kernel>
