<%inherit file='base'/>

struct kfunargs
{
    union { void (*fun)(void *); void (*fun_blks)(int, void *) };
    void *args;
    int nblocks;
};

void run_kernels(int off, int n, const struct kfunargs *kfa)
{
    for (int i = off; i < off + n; i++)
    {
        if (kfa[i].nblocks == -1)
            kfa[i].fun(kfa[i].args);
        else
        {
            #pragma omp parallel for ${schedule}
            for (int blk = 0; blk < kfa[i].nblocks; blk++)
                kfa[i].fun_blks(blk, kfa[i].args);
        }
    }
}
