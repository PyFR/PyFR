<%inherit file='base'/>

struct kargs
{
    void (*exec)(void *, const fpdtype_t *, fpdtype_t *);
    void *blockk;
    const fpdtype_t *b;
    ixdtype_t bblocksz;
    fpdtype_t *c;
    ixdtype_t cblocksz;
};

void ${kname}(ixdtype_t ib, const struct kargs *args, int _disp_mask)
{
    args->exec(args->blockk,
               args->b + ((_disp_mask & 1) ? 0 : ib*args->bblocksz),
               args->c + ((_disp_mask & 2) ? 0 : ib*args->cblocksz));
}
