# -*- coding: utf-8 -*-
<%inherit file='base'/>

// libxsmm prototype
typedef void (*libxsmm_xfsspmdm_execute)(void *, const fpdtype_t *,
                                         fpdtype_t *);

// gimmik prototype
typedef void (*gimmik_execute)(int, const fpdtype_t *, int, fpdtype_t *, int);

void
% if lib == 'xsmm':
batch_gemm(libxsmm_xfsspmdm_execute exec, void *blockk,
% else:
batch_gemm(gimmik_execute exec,
% endif
           int n, int nbcol,
           const fpdtype_t *b, int bblocksz, fpdtype_t *c, int cblocksz)
{
    #pragma omp parallel for
    for (int ib = 0; ib < n / nbcol; ib++)
        % if lib == 'xsmm':
        exec(blockk, b + ib*bblocksz, c + ib*cblocksz);
        % else:
        exec(nbcol, b + ib*bblocksz, nbcol, c + ib*cblocksz, nbcol);
        % endif
}
