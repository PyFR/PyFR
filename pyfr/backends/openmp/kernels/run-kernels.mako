<%inherit file='base'/>

#include <alloca.h>
#include <string.h>

typedef struct
{
    void (*fun)(void *);
    void *args;
} regular_t;

typedef struct
{
    int nkerns, nblocks, allocsz, nsubs;
    int *argsubs;
    struct
    {
        void (*fun)(int, void *, int);
        void *args;
        int argmask, argsz;
    } *kernels;
} block_group_t;

typedef struct
{
    enum { KTYPE_REGULAR, KTYPE_BLOCK_GROUP } ktype;
    union { regular_t regular; block_group_t block_group; };
} kfunargs_t;

void run_kernels(int n, const kfunargs_t *kfa)
{
    // Loop over each kernel or block-group thereof
    for (int i = 0; i < n; i++)
    {
        if (kfa[i].ktype == KTYPE_REGULAR)
            kfa[i].regular.fun(kfa[i].regular.args);
        else
        {
            const block_group_t bg = kfa[i].block_group;

            #pragma omp parallel
            {
                // Make local copies of the argument structures
                char ***kargs = alloca(sizeof(char **)*bg.nkerns);
                for (int j = 0; j < bg.nkerns; j++)
                {
                    kargs[j] = alloca(bg.kernels[j].argsz);
                    memcpy(kargs[j], bg.kernels[j].args, bg.kernels[j].argsz);
                }

                // Allocate any requested local storage
                char *lmem = bg.allocsz ? aligned_alloc(64, bg.allocsz) : NULL;
                int *as = bg.argsubs;

                // Apply any argument substitutions
                for (int j = 0; j < bg.nsubs; j++)
                    *(kargs[as[3*j + 0]] + as[3*j + 1]) = lmem + as[3*j + 2];

                #pragma omp for ${schedule}
                for (int blk = 0; blk < bg.nblocks; blk++)
                    for (int j = 0; j < bg.nkerns; j++)
                        bg.kernels[j].fun(blk, kargs[j], bg.kernels[j].argmask);

                if (lmem)
                    free(lmem);
            }
        }
    }
}
