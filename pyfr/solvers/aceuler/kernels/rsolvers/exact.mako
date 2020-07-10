# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.aceuler.kernels.flux1d'/>

<% zeta = c['ac-zeta'] %>
<% rzeta = 1./c['ac-zeta'] %>
<% kmax = 5 %>

// Ininitail guess for ustar from HLL
<%pyfr:macro name='init_ustar' params='ql,al,qr,ar,us'>
    fpdtype_t sl = ql[1] - al;
    fpdtype_t sr = qr[1] + ar;

    us = (${rzeta}*sl*sr*(ql[0] - qr[0]) + (qr[1]*sl - ql[1]*sr))/(sl-sr);

</%pyfr:macro>

// Calculate components for Newton iterations
<%pyfr:macro name='newton_parts' params='ul,al,ur,ar,us,fl,fr,fd'>
    fpdtype_t as = sqrt(us*us + ${zeta});
    fpdtype_t fdl, fdr;

    if(ul+al <= us+as){ // Left Raefaction
        fl  = 0.5*(${zeta}*log((ul+al)/(us+as)) + (ul*al - us*as) + (ul*ul - us*us));
        fdl = -(us + as);
    }
    else{ // Left Shock
        fpdtype_t e = us + ul;
	fpdtype_t q = sqrt(e*e + ${4.*zeta});
	fl  = -0.5*(us - ul)*(e + q);
	fdl = -0.5*(e + q + (ul - us)*(1. + e/q));
    }
    
    if(ur+ar <= us+as){ // Right Raefaction
        fr  = 0.5*(${zeta}*log((us+as)/(ur+ar)) + (us*as - ur*ar) + (ur*ur - us*us));
        fdr = as - us;
    }
    else{ // Right Shock
        fpdtype_t e = us + ur;
	fpdtype_t q = sqrt(e*e + ${4.*zeta});
	fr  = -0.5*(us - ur)*(e - q);
	fdr = -0.5*(e + q + (us - ur)*(1. + e/q));
    }
    fd = fdr + fdl;

</%pyfr:macro>

<%pyfr:macro name='riemann_decision' params='ql,al,qr,ar,ps,us,as,qs'>
    qs[0] = ps;
    qs[1] = us;

    if(us >= 0.){
        if(ul+al <= us+as){ // Left Rarefaction
% for i in range(ndims-1):
            qs[${i+2}] = ql[${i+2}]*(as/al)*(ql[1] + sqrt(2.*ql[1]*ql[1] + ${zeta}))/
                                            (us    + sqrt(2.*us   *us    + ${zeta}));
% endfor
        }
        else{ // Left Shock
            fpdtype_t sl = us + ql[1] + (ps - ql[0])/(us - ql[1]);
% for i in range(ndims-1):	    
            qs[${i+2}] = ql[${i+2}]*(sl - ql[1])/(sl - us);
% endfor
        }
    }
    else{
        if(ur+ar <= us+as){ // Right Rarefaction
% for i in range(ndims-1):
            qs[${i+2}] = qr[${i+2}]*(ar/as)*(us    + sqrt(2.*us   *us    + ${zeta}))/
                                            (qr[1] + sqrt(2.*qr[1]*qr[1] + ${zeta}));
% endfor
        }
        else{ // Right Shock
            fpdtype_t sr = us + qr[1] + (ps - qr[0])/(us - qr[1]);
% for i in range(ndims-1):	    
            qs[${i+2}] = qr[${i+2}]*(sr - qr[1])/(sr - us);
% endfor
        }
    }

</%pyfr:macro>

<%pyfr:macro name='rsolve_t1d' params='ql, qr, nf'>
    fpdtype_t us, qs[${nvars}];
    fpdtype_t ul = ql[1], ur = qr[1];
    fpdtype_t fl,fr,fd;

    // Pressure jump
    fpdtype_t dp = ql[0] - qr[0];

    // ACM speed of sound
    fpdtype_t al = sqrt(ul*ul + ${zeta});
    fpdtype_t ar = sqrt(ur*ur + ${zeta});

    ${pyfr.expand('init_ustar','ql','al','qr','ar','us')};
    
% for k in range(kmax):
    ${pyfr.expand('newton_parts','ul','al','ur','ar','us','fl','fr','fd')};
    us = us - (dp + fl - fr)/fd;
% endfor

    // Calculate some star properties
    fpdtype_t as = sqrt(us*us + ${zeta});
    fpdtype_t ps = ql[0] + fl;

    // Calculate v* and w*
    ${pyfr.expand('riemann_decision','ql','al','qr','ar','ps','us','as','qs')};

    // Final produce common flux
    ${pyfr.expand('inviscid_1dflux', 'qs', 'nf')};

</%pyfr:macro>

<%include file='pyfr.solvers.aceuler.kernels.rsolvers.rsolve_trans'/>