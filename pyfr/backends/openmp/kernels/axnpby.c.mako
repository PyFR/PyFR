# -*- coding: utf-8 -*-

<%namespace name='util' module='pyfr.backends.openmp.makoutil' />
<%include file='common.h.mako' />

void
axnpby(int n, ${dtype} *restrict y, ${dtype} beta,
       ${', '.join('const {0} *restrict x{1}, {0} a{1}'.format(dtype, i)
                   for i in range(n))})
{
    ASSUME_ALIGNED(y);
% for i in range(n):
    ASSUME_ALIGNED(x${i});
% endfor

    for (int i = 0; i < n; i++)
    {
        ${dtype} axn = ${util.dot('a{0}', 'x{0}[i]', len='n')};

        if (beta == 0.0)
            y[i] = axn;
        else if (beta == 1.0)
            y[i] += axn;
        else
            y[i] = beta*y[i] + axn;
    }
}
