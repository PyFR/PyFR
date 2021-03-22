# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <string.h>

void
par_memcpy(char *dst, int dbbytes, const char *src, int sbbytes, int bnbytes,
           int nblocks)
{
    #pragma omp parallel for
    for (int ib = 0; ib < nblocks; ib++)
        memcpy(dst + dbbytes*ib, src + sbbytes*ib, bnbytes);
}
