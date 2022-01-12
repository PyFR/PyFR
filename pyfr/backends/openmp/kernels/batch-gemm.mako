# -*- coding: utf-8 -*-
<%inherit file='base'/>

struct kargs
{
% if lib == 'xsmm':
    void (*exec)(void *, const fpdtype_t *, fpdtype_t *);
    void *blockk;
% else:
    void (*exec)(int, const fpdtype_t *, int, fpdtype_t *, int);
    int bldim;
% endif
    int nblocks;
    const fpdtype_t *b;
    int bblocksz;
    fpdtype_t *c;
    int cblocksz;
};

void batch_gemm(const struct kargs *restrict args)
{
    #pragma omp parallel for
    for (int ib = 0; ib < args->nblocks; ib++)
    % if lib == 'xsmm':
        args->exec(args->blockk,
                   args->b + ib*args->bblocksz,
                   args->c + ib*args->cblocksz);
    % else:
        args->exec(args->bldim,
                   args->b + ib*args->bblocksz, args->bldim,
                   args->c + ib*args->cblocksz, args->bldim);
    % endif
}
