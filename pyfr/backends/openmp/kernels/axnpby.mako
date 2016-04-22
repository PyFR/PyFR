# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

static void
axnpby_inner(int n, fpdtype_t *__restrict__ y, fpdtype_t beta,
             ${', '.join('const fpdtype_t *__restrict__ x{0}, '
                         'fpdtype_t a{0}'.format(i) for i in range(n))})
{
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

void
axnpby(int n, fpdtype_t *__restrict__ y, fpdtype_t beta,
       ${', '.join('const fpdtype_t *__restrict__ x{0}, '
                   'fpdtype_t a{0}'.format(i) for i in range(n))})
{
    #pragma omp parallel
    {
        int begin, end;
        loop_sched_1d(n, PYFR_ALIGN_BYTES / sizeof(fpdtype_t), &begin, &end);

        axnpby_inner(end - begin, y + begin, beta,
                     ${', '.join('x{0} + begin, a{0}'.format(i)
                                 for i in range(n))});
    }
}
