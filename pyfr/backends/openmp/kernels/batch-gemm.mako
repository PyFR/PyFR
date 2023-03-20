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

void batch_gemm(int ib, const struct kargs *args)
{
    args->exec(args->blockk,
               args->b + ib*args->bblocksz,
               args->c + ib*args->cblocksz);
}
