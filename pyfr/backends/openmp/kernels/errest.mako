# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

fpdtype_t
errest(int n, fpdtype_t *__restrict__ x, fpdtype_t *__restrict__ y,
       fpdtype_t *__restrict__ z, fpdtype_t atol, fpdtype_t rtol)
{
    fpdtype_t sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++)
        sum += pow(x[i]/(atol + rtol*max(fabs(y[i]), fabs(z[i]))), 2);

    return sum;
}
