<%inherit file='base'/>

struct kargs
{
    void (*exec)(void *, const fpdtype_t *, fpdtype_t *);
    void *blockk;
    const fpdtype_t *b;
    int bblocksz;
    fpdtype_t *c;
    int cblocksz;
};

void batch_gemm(int ib, const struct kargs *args, int _disp_mask)
{
    args->exec(args->blockk,
               args->b + ((_disp_mask & 1) ? 0 : ib*args->bblocksz),
               args->c + ((_disp_mask & 2) ? 0 : ib*args->cblocksz));
}
