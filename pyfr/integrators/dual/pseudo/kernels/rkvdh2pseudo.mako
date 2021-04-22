# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='rkvdh2pseudo' ndim='2'
              dtau='in fpdtype_t[${str(nvars)}]'
              r1='inout fpdtype_t[${str(nvars)}]'
              r2='inout fpdtype_t[${str(nvars)}]'
              rold='in fpdtype_t[${str(nvars)}]'
              rerr='inout fpdtype_t[${str(nvars)}]'>
% for j in range(nvars):
    r2[${j}] *= dtau[${j}];
% if errest and stage == 0:
    rerr[${j}] = ${e[stage]}*r2[${j}];
% elif errest:
    rerr[${j}] += ${e[stage]}*r2[${j}];
% endif

% if stage == 0:
    r1[${j}] = rold[${j}] + ${a[stage]}*r2[${j}];
    r2[${j}] = ${b[stage] - a[stage]}*r2[${j}] + r1[${j}];
% elif stage < nstages - 1:
    r1[${j}] += ${a[stage]}*r2[${j}];
    r2[${j}] = ${b[stage] - a[stage]}*r2[${j}] + r1[${j}];
% else:
    r1[${j}] += ${b[stage]}*r2[${j}];
% endif
% endfor
</%pyfr:kernel>
