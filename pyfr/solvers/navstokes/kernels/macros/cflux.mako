<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.baseadvecdiff.kernels.macros.artvisc'/>
<%include file='pyfr.solvers.euler.kernels.macros.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.macros.viscous'/>

## The LDG flux weights are the common solution weights mirrored
<% wl, wr, tau = c['ldg-cr'], c['ldg-cl'], c['ldg-tau'] %>

<%pyfr:macro name='cflux' params='ul, ur, gradul, gradur, artvisc, n, fn'>
    // Perform the Riemann solve
    fpdtype_t ficomm[${nvars}], fvcomm;
    ${pyfr.expand('rsolve', 'ul', 'ur', 'n', 'ficomm')};

% if wl != 0:
    fpdtype_t fvl[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ul', 'gradul', 'fvl')};
    ${pyfr.expand('artificial_viscosity_add', 'gradul', 'fvl', 'artvisc')};
% endif

% if wr != 0:
    fpdtype_t fvr[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ur', 'gradur', 'fvr')};
    ${pyfr.expand('artificial_viscosity_add', 'gradur', 'fvr', 'artvisc')};
% endif

% for i in range(nvars):
% if wl == 0:
    fvcomm = ${' + '.join(f'n[{j}]*fvr[{j}][{i}]' for j in range(ndims))};
% elif wr == 0:
    fvcomm = ${' + '.join(f'n[{j}]*fvl[{j}][{i}]' for j in range(ndims))};
% else:
    fvcomm = ${wl}*(${' + '.join(f'n[{j}]*fvl[{j}][{i}]'
                                 for j in range(ndims))})
           + ${wr}*(${' + '.join(f'n[{j}]*fvr[{j}][{i}]'
                                 for j in range(ndims))});
% endif
% if tau != 0.0:
    fvcomm += ${tau}*(ul[${i}] - ur[${i}]);
% endif

    fn[${i}] = ficomm[${i}] + fvcomm;
% endfor
</%pyfr:macro>
