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
batch_gemm(gimmik_execute exec, int bldim,
% endif
           int nblocks,
           const fpdtype_t *b, int bblocksz, fpdtype_t *c, int cblocksz)
{
    #pragma omp parallel for
    for (int ib = 0; ib < nblocks; ib++)
    % if lib == 'xsmm':
        exec(blockk, b + ib*bblocksz, c + ib*cblocksz);
    % else:
        exec(bldim, b + ib*bblocksz, bldim, c + ib*cblocksz, bldim);
    % endif
}
