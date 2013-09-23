# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
axnpby(int n, fpdtype_t *__restrict__ y, fpdtype_t beta,
       ${', '.join('const fpdtype_t *__restrict__ x{0}, '
                   'fpdtype_t a{0}'.format(i) for i in range(n))})
{
    PYFR_ALIGNED(y);
% for i in range(n):
    PYFR_ALIGNED(x${i});
% endfor

    for (int i = 0; i < n; i++)
    {
        fpdtype_t axn = ${pyfr.dot('a{j}', 'x{j}[i]', j=n)};

        if (beta == 0.0)
            y[i] = axn;
        else if (beta == 1.0)
            y[i] += axn;
        else
            y[i] = beta*y[i] + axn;
    }
}
