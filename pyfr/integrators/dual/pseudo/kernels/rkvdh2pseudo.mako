# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='rkvdh2pseudo' ndim='2'
              dtau='in fpdtype_t[${str(nvars)}]'
              r1='inout fpdtype_t[${str(nvars)}]'
              r2='inout fpdtype_t[${str(nvars)}]'
              rold='in fpdtype_t[${str(nvars)}]'
              rerr='inout fpdtype_t[${str(nvars)}]'>
    fpdtype_t tmpr1[] = ${pyfr.array('r1[{j}]', j=nvars)};
    fpdtype_t tmpr2[] = ${pyfr.array('dtau[{j}]*r2[{j}]', j=nvars)};
% if errest and stage > 0:
    fpdtype_t tmprerr[] = ${pyfr.array('rerr[{j}]', j=nvars)};
% endif

% for j in range(nvars):
% if errest and stage == 0:
    rerr[${j}] = ${e[stage]}*tmpr2[${j}];
% elif errest:
    rerr[${j}] = tmprerr[${j}] + ${e[stage]}*tmpr2[${j}];
% endif

% if stage == 0:
    r1[${j}] = rold[${j}] + ${a[stage]}*tmpr2[${j}];
    r2[${j}] = rold[${j}] + ${b[stage]}*tmpr2[${j}];
% elif stage < nstages - 1:
    r1[${j}] = tmpr1[${j}] + ${a[stage]}*tmpr2[${j}];
    r2[${j}] = tmpr1[${j}] + ${b[stage]}*tmpr2[${j}];
% else:
    r1[${j}] = tmpr1[${j}] + ${b[stage]}*tmpr2[${j}];
% endif
% endfor
</%pyfr:kernel>
