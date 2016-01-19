# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

static PYFR_NOINLINE void
axnpby_inner(int n,
             ${', '.join('fpdtype_t *__restrict__ x{0}, '
                         'fpdtype_t a{0}'.format(i) for i in range(nv))})
{
    for (int i = 0; i < n; i++)
    {
        fpdtype_t axn = ${pyfr.dot('a{j}', 'x{j}[i]', j=(1, nv))};

        if (a0 == 0.0)
            x0[i] = axn;
        else if (a0 == 1.0)
            x0[i] += axn;
        else
            x0[i] = a0*x0[i] + axn;
    }
}

void
axnpby(long *n,
       ${', '.join('fpdtype_t **x{0}'.format(i) for i in range(nv))},
       ${', '.join('double *a{0}'.format(i) for i in range(nv))})
{
    #pragma omp parallel
    {
        int begin, end;
        loop_sched_1d(*n, PYFR_ALIGN_BYTES / sizeof(fpdtype_t), &begin, &end);

        axnpby_inner(end - begin,
                     ${', '.join('*x{0} + begin, *a{0}'.format(i)
                                 for i in range(nv))});
    }
}
