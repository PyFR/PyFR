<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.navstokes.kernels.macros.cflux'/>

<%pyfr:kernel name='mpicflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              ur='inout mpi fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              gradur='in mpi fpdtype_t[${str(ndims)}][${str(nvars)}]'
              artvisc='in view fpdtype_t'
              nl='in fpdtype_t[${str(ndims)}]'>
    fpdtype_t mag_nl = sqrt(${pyfr.dot('nl[{i}]', i=ndims)});
    fpdtype_t norm_nl[] = ${pyfr.array('(1 / mag_nl)*nl[{i}]', i=ndims)};

    // Compute the common normal flux
    fpdtype_t fn[${nvars}];
    ${pyfr.expand('cflux', 'ul', 'ur', 'gradul', 'gradur', 'artvisc',
                  'norm_nl', 'fn')};

    // Scale and write out the common normal flux
% for i in range(nvars):
    ul[${i}] = mag_nl*fn[${i}];
% endfor
</%pyfr:kernel>
