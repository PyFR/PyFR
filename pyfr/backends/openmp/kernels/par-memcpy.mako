# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
par_memcpy(fpdtype_t *dst, int dblocksz, const fpdtype_t *src, int sblocksz,
           int datasz, int n, int nbcol)
{
    int nblocks = n/nbcol;
    #pragma omp parallel for
    for (int ib = 0; ib < nblocks; ib++)
    {
        memcpy(dst + dblocksz*ib, src + sblocksz*ib, datasz);
    }
}
