# -*- coding: utf-8 -*-
<%inherit file='base'/>

// libxsmm prototype
typedef void (*libxsmm_mm_t)(const fpdtype_t *,
                             const fpdtype_t *,
                             fpdtype_t *);

void
par_xsmm(libxsmm_mm_t xsmm, int n, int nblock,
         const fpdtype_t *a, const fpdtype_t *b, fpdtype_t *c)
{
    #pragma omp parallel for
    for (int i = 0; i < n; i += nblock)
        xsmm(b + i, a, c + i);
}
