# -*- coding: utf-8 -*-
<%inherit file='base'/>

// libxsmm prototype
typedef void (*libxsmm_xfsspmdm_execute)(void *,
                                         const fpdtype_t *,
                                         fpdtype_t *);

void
par_xsmm(libxsmm_xfsspmdm_execute exec, void *blockk, void *cleank,
         int n, int nblock, const fpdtype_t *b, fpdtype_t *c,
         int nbcol, int bblocksz, int cblocksz)
{
    int rem = nbcol % nblock;
    int nci = nbcol / nblock;
    int nblocks = n/nbcol;
    #pragma omp parallel for
    for (int ib = 0; ib < nblocks; ib++)
    {
        for (int i = 0; i < nci; i++)
            exec(blockk, b + ib*bblocksz + i*nblock, c + ib*cblocksz + i*nblock);

        if (rem != 0)
            exec(cleank, b + ib*bblocksz + nci*nblock, c + ib*cblocksz + nci*nblock);
    }
}
