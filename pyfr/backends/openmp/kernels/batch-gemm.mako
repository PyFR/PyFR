# -*- coding: utf-8 -*-
<%inherit file='base'/>

#include <stdalign.h>

struct kargs
{
    void (*exec)(void *, const fpdtype_t *, fpdtype_t *);
    int nblocks;
    const fpdtype_t *b;
    int bblocksz;
    fpdtype_t *c;
    int cblocksz;
    void *blockk[];
};

void batch_gemm(const struct kargs *restrict args)
{
    #pragma omp parallel for ${schedule}
    for (int ib = 0; ib < args->nblocks; ib++)
    {
      % if nfac == 2:
        fpdtype_t alignas(64) buf0[${blocksz}];
      % elif nfac >= 3:
        fpdtype_t alignas(64) buf0[${blocksz}], buf1[{blocksz}];
      % endif
      % for i in range(nfac):
        % if loop.first and loop.last:
        args->exec(args->blockk[0],
                   args->b + ib*args->bblocksz,
                   args->c + ib*args->cblocksz);
        % elif loop.first:
        args->exec(args->blockk[0], args->b + ib*args->bblocksz, buf0);
        % elif loop.last:
        args->exec(args->blockk[${nfac-1}], buf${(i + 1) % 2},
                   args->c + ib*args->cblocksz);
        % else:
        args->exec(args->blockk[${i}], buf${(i + 1) % 2}, buf${i % 2});
        % endif
      % endfor
    }
}
