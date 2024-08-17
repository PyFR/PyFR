<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.wmles.kernels.sgs.smagorinsky'/>
<%include file='pyfr.solvers.wmles.kernels.bcs.${bctype}'/>

% if bccfluxstate:
<%include file='pyfr.solvers.wmles.kernels.bcs.${bccfluxstate}'/>
% endif

<%pyfr:kernel name='bccflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              artviscl='in view fpdtype_t'
              nl='in fpdtype_t[${str(ndims)}]'>
    fpdtype_t mag_nl = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_nl[] = ${pyfr.array('(1 / mag_nl)*nl[{i}]', i=ndims)};

    ${pyfr.expand('bc_common_flux_state', 'ul', 'gradul', 'artviscl', 'norm_nl', 'mag_nl')};
</%pyfr:kernel>
