<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.navstokes.kernels.macros.bcflux'/>

<%pyfr:kernel name='bccflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              artvisc='in view fpdtype_t'
              nl='in fpdtype_t[${str(ndims)}]'>
    fpdtype_t mag_nl = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_nl[] = ${pyfr.array('(1 / mag_nl)*nl[{i}]', i=ndims)};

    // Compute the boundary common normal flux
    fpdtype_t fn[${nvars}];
    ${pyfr.expand('bcflux', 'ul', 'gradul', 'artvisc', 'norm_nl', 'mag_nl',
                  'fn')};

% for i in range(nvars):
    ul[${i}] = fn[${i}];
% endfor
</%pyfr:kernel>
