<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='precondscale' ndim='1'
              J='inout fpdtype_t[${str(bsize)}][${str(bsize)}]'
              gamma='scalar fpdtype_t'>
    for (int i = 0; i < ${bsize}; i++)
        for (int j = 0; j < ${bsize}; j++)
            J[i][j] *= -gamma;

    for (int i = 0; i < ${bsize}; i++)
        J[i][i] += 1;
</%pyfr:kernel>
