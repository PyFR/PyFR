# -*- coding: utf-8 -*-
<%inherit file='base'/>

// gimmik prototype
typedef void (*gimmik_execute)(int, const fpdtype_t *, int, fpdtype_t *, int);

void
par_gimmik(gimmik_execute g_exec, int n, int nbcol,
           const fpdtype_t *b, int bblocksz, fpdtype_t *c, int cblocksz)
{
    int nblocks = n/nbcol;
    #pragma omp parallel for
    for (int ib = 0; ib < nblocks; ib++)
    {
        g_exec(nbcol, b + ib*bblocksz, nbcol, c + ib*cblocksz, nbcol);
    }
}
