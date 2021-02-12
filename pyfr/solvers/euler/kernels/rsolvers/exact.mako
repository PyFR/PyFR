# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.primitives'/>

<% gmrtg = (c['gamma'] - 1)/(2*c['gamma']) %>
<% gprtg = (c['gamma'] + 1)/(2*c['gamma']) %>
<% tgrgm = (2*c['gamma'])/(c['gamma'] - 1) %>
<% tgrgp = (2*c['gamma'])/(c['gamma'] + 1) %>
<% trgm = 2/(c['gamma'] - 1) %>
<% trgp = 2/(c['gamma'] + 1) %>
<% gmrgp = (c['gamma'] - 1)/(c['gamma'] + 1) %>
<% hgm = 0.5*(c['gamma'] - 1) %>
<% rgm = 1/(c['gamma'] - 1) %>
<% gamma = c['gamma'] %>

// Initial guess for pressure
<%pyfr:macro name='init_p' params='rl, vl, pl, cl, rr, vr, pr, cr, p0'>
    fpdtype_t bpv = max(0, 0.5*(pl + pr) + 0.125*(vl[0] - vr[0])*(rl + rr)*(cl + cr));
    fpdtype_t pmin = min(pl, pr);
    fpdtype_t pmax = max(pl, pr);
    fpdtype_t rpmax = pmax / pmin;

    if (rpmax <= 2 && pmin <= bpv && bpv <= pmax)
    {
        p0 = bpv;
    }
    // Two-rarefaction Riemann solve
    else if (bpv < pmin)
    {   
        fpdtype_t rcl = 1 / cl; 
        fpdtype_t rcr = 1 / cr;

        fpdtype_t pre = pow(pl / pr, ${gmrtg});
        fpdtype_t um  = (pre*vl[0]*rcl + vr[0]*rcr + ${trgm}*(pre - 1)) / (pre*rcl + 1*rcr);

        fpdtype_t ptl = 1 - ${hgm}*(um - vl[0])*rcl;
        fpdtype_t ptr = 1 + ${hgm}*(um - vr[0])*rcr;

        p0 = 0.5*(pl*pow(ptl, ${tgrgm}) + pr*pow(ptr, ${tgrgm}));
    }
    // Two-shock Riemann solve
    else
    {   
        fpdtype_t gl = sqrt((${trgp}/rl) / (${gmrgp}*pl + bpv));
        fpdtype_t gr = sqrt((${trgp}/rr) / (${gmrgp}*pr + bpv));
        p0 = (gl*pl + gr*pr - (vr[0] - vl[0])) / (gl + gr);
    }
</%pyfr:macro>

// Star Flux, assuming covolume = 0. See Toro 2009 Eq.(4.86-4.87)
<%pyfr:macro name='star_flux' params='p, ps, rs, cs, f, fd'>
    if (p <= ps)
    {
       fpdtype_t pr = p/ps;
       f  = ${trgm}*cs*(pow(pr, ${gmrtg}) - 1);
       fd = pow(pr, ${-gprtg}) / (rs*cs);
    }
    else
    {
       fpdtype_t as = ${trgp}/rs;
       fpdtype_t bs = ${gmrgp}*ps;
       fpdtype_t sapb = sqrt(as / (p + bs));
       f  = (p - ps)*sapb;
       fd = (1 - 0.5*(p - ps) / (p + bs))*sapb;
    }
</%pyfr:macro>

// Primitive to inviscid flux, w = [density, v_1,..., v_ndims, p]^T
<%pyfr:macro name='primitive_1dflux' params='w, f'>
    fpdtype_t invrho = 1 / w[0];

    // Compute the velocities
    fpdtype_t rhov[${ndims}];
% for i in range(ndims):
    rhov[${i}] = w[0]*w[${i + 1}];
% endfor

    // Compute the Energy
    fpdtype_t E = w[${nvars - 1}]*${rgm} + 0.5*invrho*(${pyfr.dot('rhov[{i}]', i=ndims)});

    // Density and energy fluxes
    f[0] = rhov[0];
    f[${nvars - 1}] = (E + w[${nvars - 1}])*w[1];

    // Momentum fluxes
    f[1]= rhov[0]*w[1] + w[${nvars - 1}];
% for i in range(1, ndims):
    f[${i + 1}]= rhov[0]*w[${i + 1}];
% endfor
</%pyfr:macro>

// Exact solve solution decision tree
<% switch = 0.0 %>
<%pyfr:macro name='riemann_decision' params='rl, vl, pl, cl, rr, vr, pr, cr, us, p0, w0'>
    if (${switch} <= us)
    {
% for i in range(ndims-1):
        w0[${i + 2}] = vl[${i + 1}];
% endfor
        if (p0 <= pl)
        {
            if (${switch} <= (vl[0] - cl))
            {
                w0[0] = rl;
                w0[1] = vl[0];
                w0[${nvars - 1}] = pl;
            }
            else
            {
                fpdtype_t cml = cl*pow(p0/pl, ${gmrtg});
                if (${switch} > (us - cml))
                {
                    w0[0] = rl*pow(p0/pl, ${1/gamma});
                    w0[1] = us;
                    w0[${nvars - 1}] = p0;
                }
                else
                {
                    fpdtype_t c = ${trgp}*(cl + ${hgm}*(vl[0] - ${switch}));
                    w0[0] = rl*pow(c/cl, ${trgm});
                    w0[1] = ${trgp}*(cl + ${hgm}*vl[0] + ${switch});
                    w0[${nvars - 1}] = pl*pow(c/cl, ${tgrgm});
                }
            }
        }
        else
        {
            fpdtype_t p0p = p0 / pl;
            fpdtype_t sl = vl[0] - cl*sqrt(${gprtg}*p0p + ${gmrtg});
            if (${switch} <= sl)
            {
                w0[0] = rl;
                w0[1] = vl[0];
                w0[${nvars - 1}] = pl;
            }
            else
            {
                w0[0] = rl*(p0p + ${gmrgp}) / (p0p*${gmrgp} + 1);
                w0[1] = us;
                w0[${nvars - 1}] = p0;
            }
        }
    }
    else
    {
% for i in range(ndims-1):
        w0[${i + 2}] = vr[${i + 1}];
% endfor
        if (p0 > pr)
        {
            fpdtype_t p0p = p0 / pr;
            fpdtype_t sr = vr[0] + cr*sqrt(${gprtg}*p0p + ${gmrtg});
            if (${switch} >= sr)
            {
                w0[0] = rr;
                w0[1] = vr[0];
                w0[${nvars - 1}] = pr;
            }
            else
            {
                w0[0] = rr*(p0p + ${gmrgp}) / (p0p*${gmrgp} + 1);
                w0[1] = us;
                w0[${nvars - 1}] = p0;
            }
        }
        else
        {
            if (${switch} >= vr[0] + cr)
            {
                w0[0] = rr;
                w0[1] = vr[0];
                w0[${nvars - 1}] = pr;
            }
            else
            {
                fpdtype_t p0p = p0 / pr;
                fpdtype_t cmr = cr*pow(p0p, ${gmrtg});
                if (${switch} <= (us + cmr))
                {
                    w0[0] = rr*pow(p0p, ${1/gamma});
                    w0[1] = us;
                    w0[${nvars - 1}] = p0;
                }
                else
                {
                    fpdtype_t c = ${trgp}*(cr - ${hgm}*(vr[0] - ${switch}));
                    w0[0] = rr*pow(c/cr, ${trgm});
                    w0[1] = ${trgp}*(-cr + ${hgm}*vr[0] + ${switch});
                    w0[${nvars - 1}] = pr*pow(c/cr, ${tgrgm});
                }
            }
        }
    }
</%pyfr:macro>

// Godunov exact Riemann solver
<% kmax = 3 %>
<% pmin = 0.00001 %>
<%pyfr:macro name='rsolve_t1d' params='ul, ur, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t vl[${ndims}],vr[${ndims}];
    fpdtype_t pl,pr,p0,p1;
    fpdtype_t fsl,fsr,fdl,fdr;
    fpdtype_t w0[${nvars}];

    ${pyfr.expand('inviscid_prim', 'ul', 'pl', 'vl')};
    ${pyfr.expand('inviscid_prim', 'ur', 'pr', 'vr')};

    // Calculate Left/Right sound speeds
    fpdtype_t cl = sqrt(${c['gamma']}*pl / ul[0]);
    fpdtype_t cr = sqrt(${c['gamma']}*pr / ur[0]);

    // Inital pressure guess
    fpdtype_t rl = ul[0];
    fpdtype_t rr = ur[0];
    ${pyfr.expand('init_p', 'rl', 'vl', 'pl', 'cl',
                            'rr', 'vr', 'pr', 'cr', 'p0')};
    fpdtype_t ud = vr[0] - vl[0];

    // Newton Iterations
% for k in range(kmax):
    ${pyfr.expand('star_flux', 'p0', 'pl', 'rl', 'cl', 'fsl', 'fdl')};
    ${pyfr.expand('star_flux', 'p0', 'pr', 'rr', 'cr', 'fsr', 'fdr')};
    p1 = p0 - (fsl + fsr + ud) / (fdl + fdr);
    p0 = (p1 < 0) ? ${pmin} : p1;
% endfor
    fpdtype_t us = 0.5*(vl[0] + vr[0] + fsr - fsl);

    // Go through Riemann solve decision tree
    ${pyfr.expand('riemann_decision', 'rl', 'vl', 'pl', 'cl',
                                      'rr', 'vr', 'pr', 'cr', 'us', 'p0', 'w0')};
    ${pyfr.expand('primitive_1dflux', 'w0', 'nf')};
</%pyfr:macro>

<%include file='pyfr.solvers.euler.kernels.rsolvers.rsolve_trans'/>
