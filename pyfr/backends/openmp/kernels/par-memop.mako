<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <string.h>

struct kargs
{
    char *dst;
% if op == 'copy':
    const char *src;
    ixdtype_t dbbytes, sbbytes, bnbytes, nblocks;
% else:
    ixdtype_t dbbytes, bnbytes, nblocks;
% endif
};

void par_${op}(const struct kargs *restrict args)
{
    #pragma omp parallel for ${schedule}
    for (ixdtype_t ib = 0; ib < args->nblocks; ib++)
% if op == 'copy':
        memcpy(args->dst + ((size_t) args->dbbytes)*ib,
               args->src + ((size_t) args->sbbytes)*ib, args->bnbytes);
% else:
        memset(args->dst + ((size_t) args->dbbytes)*ib, 0, args->bnbytes);
% endif
}
