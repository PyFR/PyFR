<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='compute_pressure' params='cons, p'>
    fpdtype_t invrho = 1.0/cons[0];
    p = ${c['gamma'] - 1}*(cons[${str(nvars - 1)}]
        - 0.5*invrho*(${pyfr.dot('cons[{i}]', i=(1, ndims + 1))}));
</%pyfr:macro>

<%pyfr:macro name='compute_sound_speed' params='rho, p, csnd'>
    csnd = sqrt(${c['gamma']}*p/rho);
</%pyfr:macro>

<%pyfr:macro name='con_to_pri' params='cons, pri'>
    fpdtype_t invrho = 1.0/cons[0];
    pri[0] = cons[0];
% for i in range(ndims):
    pri[${i + 1}] = invrho*cons[${i + 1}];
% endfor
    pri[${str(nvars - 1)}] = ${c['gamma'] - 1}*(cons[${str(nvars - 1)}]
        - 0.5*invrho*(${pyfr.dot('cons[{i}]', i=(1, ndims + 1))}));
</%pyfr:macro>

<%pyfr:macro name='grad_con_to_pri' params='cons, grad_cons, grad_pri'>
    fpdtype_t rho = cons[0], invrho = 1.0/rho;
    fpdtype_t vel[${ndims}], rhov[${ndims}];
% for i in range(ndims):
    rhov[${i}] = cons[${i + 1}];
    vel[${i}] = invrho*rhov[${i}];
% endfor
% for d in range(ndims):
    grad_pri[0][${d}] = grad_cons[${d}][0];
% endfor
% for i, d in pyfr.ndrange(ndims, ndims):
    grad_pri[${i + 1}][${d}] = invrho*(grad_cons[${d}][${i + 1}] - vel[${i}]*grad_cons[${d}][0]);
% endfor
% for d in range(ndims):
    {
        fpdtype_t term = 0;
% for i in range(ndims):
        term += vel[${i}]*grad_cons[${d}][${i + 1}] + rhov[${i}]*grad_pri[${i + 1}][${d}];
% endfor
        grad_pri[${str(nvars - 1)}][${d}] = ${c['gamma'] - 1}*(grad_cons[${d}][${str(nvars - 1)}] - 0.5*term);
    }
% endfor
</%pyfr:macro>
