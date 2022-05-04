# -*- coding: utf-8 -*-
<%inherit file='base'/>

struct kargs
{
    void (*exec)(void *, const fpdtype_t *, fpdtype_t *);
    void *blockk;
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
        args->exec(args->blockk,
                   args->b + ib*args->bblocksz,
                   args->c + ib*args->cblocksz);
}
