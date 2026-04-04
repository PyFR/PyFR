<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='applyprecond' ndim='1'
              x='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              minv='in fpdtype_t[${str(block_size)}][${str(block_size)}]'
              y='out fpdtype_t[${str(nupts)}][${str(nvars)}]'>
% if in_scale:
    const fpdtype_t _in[] = ${pyfr.carray(in_scale)};
% endif
% if out_scale:
    const fpdtype_t _out[] = ${pyfr.carray(out_scale)};
% endif

    fpdtype_t xf[${block_size}];
    for (int i = 0; i < ${nupts}; i++)
        for (int k = 0; k < ${nvars}; k++)
            xf[i*${nvars} + k] = ${'_in[k]*' if in_scale else ''}x[i][k];

    for (int i = 0; i < ${nupts}; i++)
    {
        for (int k = 0; k < ${nvars}; k++)
        {
            int row = i*${nvars} + k;
            fpdtype_t acc = 0;
            for (int col = 0; col < ${block_size}; col++)
                acc += minv[row][col]*xf[col];
            y[i][k] = ${'_out[k]*' if out_scale else ''}acc;
        }
    }
</%pyfr:kernel>
