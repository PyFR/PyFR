<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='fluidforce' ndim='1'
              u='in view fpdtype_t[${str(nupts)}][${str(nvars)}]'
              gradu='in view fpdtype_t[${str(ndims*nupts)}][${str(nvars)}]'
              wnorms='in fpdtype_t[${str(nfpts)}][${str(ndims)}]'
              rfpts='in fpdtype_t[${str(nfpts)}][${str(ndims)}]'
              pf='out fpdtype_t[${str(nout)}]'>
    fpdtype_t pf_acc[${nout}] = {};

% for fpt in range(nfpts):
    {
        // Interpolate conservative solution to fpt ${fpt}
        fpdtype_t ufpt[${nvars}] = {};
    % if viscous:
        fpdtype_t dufpt[${ndims}][${nvars}] = {};
    % endif
    % for upt, w in enumerate(m0[fpt]):
    % if w:
        {
        % for v in range(nvars):
            ufpt[${v}] += ${w}*u[${upt}][${v}];
        % endfor
    % if viscous:
        % for dd, v in pyfr.ndrange(ndims, nvars):
            dufpt[${dd}][${v}] += ${w}*gradu[${dd*nupts + upt}][${v}];
        % endfor
    % endif
        }
    % endif
    % endfor

        // Compute pressure
        fpdtype_t invrho = 1.0/ufpt[0];
        fpdtype_t p = ${c['gamma'] - 1}*(ufpt[${nvars - 1}]
            - 0.5*invrho*(${pyfr.dot('ufpt[{i}]', i=(1, ndims + 1))}));

        // Pressure force at this face point
    % if mcomp:
        fpdtype_t fp[${ndims}];
    % endif
    % for d in range(ndims):
    % if mcomp:
        pf_acc[${d}] += fp[${d}] = p*wnorms[${fpt}][${d}];
    % else:
        pf_acc[${d}] += p*wnorms[${fpt}][${d}];
    % endif
    % endfor

    % if mcomp:
        // Pressure moment: r x F_p
    % if ndims == 3:
        pf_acc[${ndims}] += rfpts[${fpt}][1]*fp[2] - rfpts[${fpt}][2]*fp[1];
        pf_acc[${ndims + 1}] += rfpts[${fpt}][2]*fp[0] - rfpts[${fpt}][0]*fp[2];
        pf_acc[${ndims + 2}] += rfpts[${fpt}][0]*fp[1] - rfpts[${fpt}][1]*fp[0];
    % else:
        pf_acc[${ndims}] += rfpts[${fpt}][0]*fp[1] - rfpts[${fpt}][1]*fp[0];
    % endif
    % endif

    % if viscous:
        // Compute velocity from conservative variables
    % for i in range(ndims):
        fpdtype_t vel${i} = invrho*ufpt[${i + 1}];
    % endfor

        // Velocity derivatives following flux.mako convention
    % for dd, i in pyfr.ndrange(ndims, ndims):
        fpdtype_t dv${i}_d${dd} = dufpt[${dd}][${i + 1}] - vel${i}*dufpt[${dd}][0];
    % endfor

        // Compute viscosity
    % if visc_corr == 'sutherland':
        fpdtype_t cpT = ${c['gamma']}*(invrho*ufpt[${nvars - 1}]
                      - 0.5*(${pyfr.dot('vel{i}', i=ndims)}));
        fpdtype_t Trat = ${1.0/c['cpTref']}*cpT;
        fpdtype_t mu_c = ${c['mu']*(c['cpTref'] + c['cpTs'])}*Trat*sqrt(Trat)
                       / (cpT + ${c['cpTs']});
    % else:
        fpdtype_t mu_c = ${c['mu']};
    % endif

        // Divergence of velocity
        fpdtype_t div_v = ${ ' + '.join(f'dv{i}_d{i}' for i in range(ndims)) };

        // Viscous traction: tau[d][k]*wnorm[k]
    % if mcomp:
        fpdtype_t fv[${ndims}];
    % endif
    % for d in range(ndims):
        {
        % for k in range(ndims):
<%
            bulk = f' + {2.0/3.0}*mu_c*invrho*div_v' if d == k else ''
%>\
            fpdtype_t tau${k} = -mu_c*invrho*(dv${d}_d${k} + dv${k}_d${d})${bulk};
        % endfor
            fpdtype_t vf = ${pyfr.dot('tau{k}', f'wnorms[{fpt}][{{k}}]', k=ndims)};
            pf_acc[${ndims + mcomp + d}] += vf;
        % if mcomp:
            fv[${d}] = vf;
        % endif
        }
    % endfor

    % if mcomp:
        // Viscous moment: r x F_v
    % if ndims == 3:
        pf_acc[${2*ndims + mcomp}] += rfpts[${fpt}][1]*fv[2] - rfpts[${fpt}][2]*fv[1];
        pf_acc[${2*ndims + mcomp + 1}] += rfpts[${fpt}][2]*fv[0] - rfpts[${fpt}][0]*fv[2];
        pf_acc[${2*ndims + mcomp + 2}] += rfpts[${fpt}][0]*fv[1] - rfpts[${fpt}][1]*fv[0];
    % else:
        pf_acc[${2*ndims + mcomp}] += rfpts[${fpt}][0]*fv[1] - rfpts[${fpt}][1]*fv[0];
    % endif
    % endif
    % endif
    }
% endfor

    // Write per-element contributions
% for i in range(nout):
    pf[${i}] = pf_acc[${i}];
% endfor
</%pyfr:kernel>
