# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <string.h>

void
par_memcpy(fpdtype_t *dst, int dblocksz, const fpdtype_t *src, int sblocksz,
           int datasz, int n, int nbcol)
{
    if (dblocksz == sblocksz)
    {
        #pragma omp parallel for
        for (int i = 0; i < n / nbcol * dblocksz; i++)
        {
            memcpy(dst + i, src + i, sizeof(fpdtype_t));
        }
    }
    else
    {
        #pragma omp parallel for
        for (int ib = 0; ib < n / nbcol; ib++)
        {
            memcpy(dst + dblocksz*ib, src + sblocksz*ib, datasz);
        }
    }
}
