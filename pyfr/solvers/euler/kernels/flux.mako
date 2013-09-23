# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

% if ndims == 2:
<%pyfr:function name='inviscid_flux'
                params='const fpdtype_t s[4], fpdtype_t f[2][4],
                        fpdtype_t* pout, fpdtype_t vout[2]'>
    fpdtype_t rho = s[0], rhou = s[1], rhov = s[2], E = s[3];

    fpdtype_t invrho = 1.0/rho;
    fpdtype_t u = invrho*rhou, v = invrho*rhov;

    // Compute the pressure
    fpdtype_t p = ${c['gamma'] - 1.0}*(E - 0.5*(rhou*u + rhov*v));

    f[0][0] = rhou;         f[1][0] = rhov;

    f[0][1] = rhou*u + p;   f[1][1] = rhov*u;
    f[0][2] = rhou*v;       f[1][2] = rhov*v + p;

    f[0][3] = (E + p)*u;    f[1][3] = (E + p)*v;

    if (pout != NULL)
    {
        *pout = p;
    }

    if (vout != NULL)
    {
        vout[0] = u; vout[1] = v;
    }
</%pyfr:function>
% elif ndims == 3:
<%pyfr:function name='inviscid_flux'
                params='const fpdtype_t s[5], fpdtype_t f[3][5],
                        fpdtype_t* pout, fpdtype_t vout[3]'>
    fpdtype_t rho = s[0], rhou = s[1], rhov = s[2], rhow = s[3], E = s[4];

    fpdtype_t invrho = 1.0/rho;
    fpdtype_t u = invrho*rhou, v = invrho*rhov, w = invrho*rhow;

    // Compute the pressure
    fpdtype_t p = ${c['gamma'] - 1.0}*(E - 0.5*(rhou*u + rhov*v + rhow*w));

    f[0][0] = rhou;         f[1][0] = rhov;         f[2][0] = rhow;

    f[0][1] = rhou*u + p;   f[1][1] = rhov*u;       f[2][1] = rhow*u;
    f[0][2] = rhou*v;       f[1][2] = rhov*v + p;   f[2][2] = rhow*v;
    f[0][3] = rhou*w;       f[1][3] = rhov*w;       f[2][3] = rhow*w + p;

    f[0][4] = (E + p)*u;    f[1][4] = (E + p)*v;    f[2][4] = (E + p)*w;

    if (pout != NULL)
    {
        *pout = p;
    }

    if (vout != NULL)
    {
        vout[0] = u; vout[1] = v; vout[2] = w;
    }
</%pyfr:function>
% endif
