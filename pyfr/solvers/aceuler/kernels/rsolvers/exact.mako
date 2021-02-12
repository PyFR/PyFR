# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.aceuler.kernels.flux'/>

<% zeta = c['ac-zeta'] %>
<% rzeta = 1/c['ac-zeta'] %>
<% kmax = 3 %>

// Ininitail guess for ustar from HLL
<%pyfr:macro name='init_ustar' params='ql, al, qr, ar, us'>
    fpdtype_t sl = ql[1] - al;
    fpdtype_t sr = qr[1] + ar;

    us = (${rzeta}*sl*sr*(ql[0] - qr[0]) + (qr[1]*sl - ql[1]*sr)) / (sl - sr);
</%pyfr:macro>

// Calculate components for Newton iterations
<%pyfr:macro name='newton_parts' params='ql, al, qr, ar, us, psl, psr, dpsl, dpsr'>
    fpdtype_t as = sqrt(us*us + ${zeta});

    if (ql[1] - al <= us - as)
    {   // Left Rarefaction
        psl  = ql[0] + 0.5*(ql[1]*(al + ql[1]) - us*(as + us) 
                     + ${zeta}*log((al + ql[1]) / (as + us)));
        dpsl = -0.5*(as + us);
    }
    else
    {   // Left Shock
        fpdtype_t e = ql[1] + us;
        fpdtype_t q = sqrt(e*e + ${4*zeta});
        psl  = ql[0] + 0.5*(ql[1] - us)*(e + q);
        dpsl = 0.5*((ql[1] - us)*e/q - q) - us;
    }

    if (us + as <= qr[1] + ar)
    {   // Right Rarefaction
        psr  = qr[0] + 0.5*(us*(as - us) - qr[1]*(ar - qr[1]) 
                     + ${zeta}*log((as + us) / (ar + qr[1])));
        dpsr = 0.5*(as - us);
    }
    else
    {   // Right Shock
        fpdtype_t e = qr[1] + us;
        fpdtype_t q = sqrt(e*e + ${4*zeta});
        psr  = qr[0] - 0.5*(us - qr[1])*(e - q);
        dpsr = 0.5*((us - qr[1])*e/q + q) - us;
    }
</%pyfr:macro>

<%pyfr:macro name='riemann_decision' params='ql, al, qr, ar, ps, us, as, qs'>

    if (0 <= us)
    {
        fpdtype_t lc;
        if (ql[1] - al <= us - as)
        {   // Left Rarefaction
            if (0 <= ql[1] - al)
            {   // Left State
                lc = 1;
                qs[0] = ql[0];
                qs[1] = ql[1];
            }
            else
            {   // Star State
                lc = (as*(ql[1] + sqrt(2*ql[1]*ql[1] + ${zeta})))
                    /(al*(us    + sqrt(2*us   *us    + ${zeta})));
                qs[0] = ps;
                qs[1] = us;
            }
        }
        else
        {   // Left Shock
            if (0 < (us - ql[1]) / (ps - ql[0]))
            {   // Left State
                lc = 1;
                qs[0] = ql[0];
                qs[1] = ql[1];
            }
            else
            {   // Star State
                fpdtype_t ssl = us + ql[1] + (ps - ql[0]) / (us - ql[1]);
                lc = (ssl - ql[1]) / (ssl - us);
                qs[0] = ps;
                qs[1] = us;
            }
        }
% for i in range(ndims - 1):
        qs[${i + 2}] = ql[${i + 2}]*lc;
% endfor
    }
    else
    {
        fpdtype_t rc;
        if (us + as <= qr[1] + ar)
        {   // Right Rarefaction
            if (0 >= qr[1] + ar)
            {   // Right State
                rc = 1;
                qs[0] = qr[0];
                qs[1] = qr[1];      
            }
            else
            {   // Star State
                rc = (ar*(us    + sqrt(2*us   *us    + ${zeta})))
                    /(as*(qr[1] + sqrt(2*qr[1]*qr[1] + ${zeta})));
                qs[0] = ps;
                qs[1] = us;
            }
        }
        else
        {   // Right Shock
            if (0 > (us - qr[1]) / (ps - qr[0]))
            {   // Right State
                rc = 1;
                qs[0] = qr[0];
                qs[1] = qr[1];
            }
            else
            {   // Star State
                fpdtype_t ssr = us + qr[1] + (ps - qr[0]) / (us - qr[1]);
                rc = (ssr - qr[1]) / (ssr - us);
                qs[0] = ps;
                qs[1] = us;
            }
        }
% for i in range(ndims - 1):      
        qs[${i + 2}] = qr[${i + 2}]*rc;
% endfor
    }

</%pyfr:macro>

<%pyfr:macro name='rsolve_t1d' params='ql, qr, nf'>
    fpdtype_t us, qs[${nvars}];
    fpdtype_t psl, dpsl;
    fpdtype_t psr, dpsr;

    // ACM speed of sound
    fpdtype_t al = sqrt(ql[1]*ql[1] + ${zeta});
    fpdtype_t ar = sqrt(qr[1]*qr[1] + ${zeta});

    ${pyfr.expand('init_ustar','ql','al','qr','ar','us')};
    
% for k in range(kmax):
    ${pyfr.expand('newton_parts','ql','al','qr','ar','us','psl','psr','dpsl','dpsr')};
    us = us - (psl - psr) / (dpsl - dpsr);
% endfor

    // Calcql[1]ate some star properties
    fpdtype_t as = sqrt(us*us + ${zeta});
    fpdtype_t ps = psr;

    // Calculate v* and w*
    ${pyfr.expand('riemann_decision','ql','al','qr','ar','ps','us','as','qs')};

    // Final produce common flux
    ${pyfr.expand('inviscid_1dflux', 'qs', 'nf')};

</%pyfr:macro>

<%include file='pyfr.solvers.aceuler.kernels.rsolvers.rsolve_trans'/>
