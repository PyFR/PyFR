# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <string.h>

void
par_memcpy(char *dst, const char *src, int n)
{
    #pragma omp parallel
    {
        int begin, end;
        loop_sched_1d(n, 1, &begin, &end);

        memcpy(dst + begin, src + begin, end - begin);
    }
}
