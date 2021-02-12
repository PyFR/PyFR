# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='rkvdh2' ndim='2'
              r1='inout fpdtype_t[${str(nvars)}]'
              r2='inout fpdtype_t[${str(nvars)}]'
              rold='out fpdtype_t[${str(nvars)}]'
              rerr='inout fpdtype_t[${str(nvars)}]'
              dt='scalar fpdtype_t'>
% for j in range(nvars):
% if errest:
% if stage == 0:
    rerr[${j}] = dt*${e[stage]}*r2[${j}];
    rold[${j}] = r1[${j}];
% else:
    rerr[${j}] += dt*${e[stage]}*r2[${j}];
% endif
% endif

% if stage < nstages - 1:
    r1[${j}] += dt*${a[stage]}*r2[${j}];
    r2[${j}] = dt*${b[stage] - a[stage]}*r2[${j}] + r1[${j}];
% else:
    r1[${j}] += dt*${b[stage]}*r2[${j}];
% endif
% endfor
</%pyfr:kernel>
