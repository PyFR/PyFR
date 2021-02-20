# -*- coding: utf-8 -*-
<%inherit file='base'/>

// libxsmm prototype
typedef void (*libxsmm_xfsspmdm_execute)(void *,
                                         const fpdtype_t *,
                                         fpdtype_t *);

void
par_xsmm(libxsmm_xfsspmdm_execute exec, void *blockk, int n, int nbcol,
         const fpdtype_t *b, int bblocksz, fpdtype_t *c, int cblocksz)
{
    #pragma omp parallel for
    for (int ib = 0; ib < n / nbcol; ib++)
    {
        exec(blockk, b + ib*bblocksz, c + ib*cblocksz);
    }
}
