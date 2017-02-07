# -*- coding: utf-8 -*-
<%inherit file='base'/>

// libxsmm prototype
typedef void (*libxsmm_xfsspmdm_execute)(void *,
                                         const fpdtype_t *,
                                         fpdtype_t *);

void
par_xsmm(libxsmm_xfsspmdm_execute exec, void *blockk, void *cleank,
         int n, int nblock,
         const fpdtype_t *b, fpdtype_t *c)
{
    int endb = n - n % nblock;

    #pragma omp parallel for
    for (int i = 0; i < endb; i += nblock)
        exec(blockk, b + i, c + i);

    if (endb != n)
        exec(cleank, b + endb, c + endb);
}
