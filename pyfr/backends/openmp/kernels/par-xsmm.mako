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
    //int endb = n - n % nblock;

    //for (int i = 0; i < endb; i += nblock)
    //    exec(blockk, b + i, c + i);

    //if (endb != n)
    //    exec(cleank, b + endb, c + endb);

    int rem = nbcol % nblock;
    int nci = nbcol / nblock;
    int lenAoAoSoA = n/nbcol;
    #pragma omp parallel for
    for (int ib = 0; ib < lenAoAoSoA; ib++)
    {
        for (int i = 0; i < nci; i++)
            exec(blockk, b + ib*bblocksz + i*nblock, c + ib*cblocksz + i*nblock);

        if (rem != 0)
            exec(cleank, b + ib*bblocksz + nci*nblock, c + ib*cblocksz + nci*nblock);
    }
    int ib = lenAoAoSoA;
    rem = (n % nbcol) % nblock;
    nci = (n % nbcol) / nblock;
    for (int i = 0; i < nci; i++)
        exec(blockk, b + ib*bblocksz + i*nblock, c + ib*cblocksz + i*nblock);

    if (rem != 0)
        exec(cleank, b + ib*bblocksz + nci*nblock, c + ib*cblocksz + nci*nblock);
}
