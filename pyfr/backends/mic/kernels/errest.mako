# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
errest(long *n, double *out,
       fpdtype_t **xp, fpdtype_t **yp, fpdtype_t **zp,
       double *atolp, double *rtolp)
{
    fpdtype_t *x = *xp, *y = *yp, *z = *zp;
    fpdtype_t atol = *atolp, rtol = *rtolp;

    fpdtype_t sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < *n; i++)
        sum += pow(x[i]/(atol + rtol*max(fabs(y[i]), fabs(z[i]))), 2);

    *out = sum;
}
