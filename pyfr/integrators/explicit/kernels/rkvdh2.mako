<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='rkvdh2' ndim='2'
              r1='inout fpdtype_t[${str(nvars)}]'
              r2='inout fpdtype_t[${str(nvars)}]'
              rold='out fpdtype_t[${str(nvars)}]'
              rerr='inout fpdtype_t[${str(nvars)}]'
              r1c='in fpdtype_t[${str(nvars)}]'
              rinc='inout fpdtype_t[${str(nvars)}]'
              routc='out fpdtype_t[${str(nvars)}]'
              dt='scalar fpdtype_t'>
    fpdtype_t tmpr1[] = ${pyfr.array('r1[{j}]', j=nvars)};
    fpdtype_t tmpr2[] = ${pyfr.array('r2[{j}]', j=nvars)};
% if errest and stage > 0:
    fpdtype_t tmprerr[] = ${pyfr.array('rerr[{j}]', j=nvars)};
% endif
% if comp_accum and stage > 0:
    fpdtype_t tmprinc[] = ${pyfr.array('rinc[{j}]', j=nvars)};
% endif

## Renormalise into the solution bank unless the step may be rejected
<% rout = 'r2' if errest else 'r1' %>
% for j in range(nvars):
% if errest and stage == 0:
    rerr[${j}] = dt*${e[stage]}*tmpr2[${j}];
% if not comp_accum:
    rold[${j}] = tmpr1[${j}];
% endif
% elif errest:
    rerr[${j}] = tmprerr[${j}] + dt*${e[stage]}*tmpr2[${j}];
% endif

% if comp_accum:
<% inc = f'tmprinc[{j}] + ' if stage > 0 else '' %>
% if stage < nstages - 1:
    rinc[${j}] = ${inc}dt*${b[stage]}*tmpr2[${j}];
<% s = f'{inc}dt*{a[stage]}*tmpr2[{j}]' %>
    ${pyfr.compadd(hi=f'tmpr1[{j}]', lo=f'r1c[{j}]', inc=s, ohi=f'r2[{j}]')}
% else:
## Add the step increment to the solution pair and renormalise it
<% s = f'{inc}dt*{b[stage]}*tmpr2[{j}]' %>
<% rj, rcj = f'{rout}[{j}]', f'routc[{j}]' %>
    ${pyfr.compadd(hi=f'tmpr1[{j}]', lo=f'r1c[{j}]', inc=s, ohi=rj, olo=rcj)}
% endif
% elif stage < nstages - 1:
    r1[${j}] = tmpr1[${j}] + dt*${a[stage]}*tmpr2[${j}];
    r2[${j}] = tmpr1[${j}] + dt*${b[stage]}*tmpr2[${j}];
% else:
    r1[${j}] = tmpr1[${j}] + dt*${b[stage]}*tmpr2[${j}];
% endif
% endfor
</%pyfr:kernel>
