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

    fpdtype_t err = 0.0;

% if norm == 'l2':
    #pragma omp parallel for reduction(+:err)
    for (int i = 0; i < *n; i++)
        err += pow(x[i]/(atol + rtol*max(fabs(y[i]), fabs(z[i]))), 2);
% else:
    #pragma omp parallel for reduction(max:err)
    for (int i = 0; i < *n; i++)
        err = max(err, pow(x[i]/(atol + rtol*max(fabs(y[i]), fabs(z[i]))), 2));
% endif

    *out = err;
}
