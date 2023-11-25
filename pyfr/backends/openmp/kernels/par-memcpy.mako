<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <string.h>

struct kargs
{
    char *dst;
    const char *src;
    ixdtype_t dbbytes, sbbytes, bnbytes, nblocks;
};

void par_memcpy(const struct kargs *restrict args)
{
    #pragma omp parallel for ${schedule}
    for (ixdtype_t ib = 0; ib < args->nblocks; ib++)
        memcpy(args->dst + ((size_t) args->dbbytes)*ib,
               args->src + ((size_t) args->sbbytes)*ib, args->bnbytes);
}
