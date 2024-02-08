<%inherit file='base'/>

struct kargs
{
    void (*exec)(void *, const fpdtype_t *, fpdtype_t *);
    void *blockk;
    void *blockk_nt;
    const fpdtype_t *b;
    ixdtype_t bblocksz;
    fpdtype_t *c;
    ixdtype_t cblocksz;
};

void batch_gemm(ixdtype_t ib, const struct kargs *args, int _disp_mask)
{
    args->exec((_disp_mask & 32) ? args->blockk : args->blockk_nt,
                args->b + ((_disp_mask & 8) ? 0 : ib*args->bblocksz),
                args->c + ((_disp_mask & 32) ? 0 : ib*args->cblocksz));
}
