<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// AoSoA macros
#define SOA_SZ ${soasz}
#define SOA_IX(a, v, nv) ((((a) / SOA_SZ)*(nv) + (v))*SOA_SZ + (a) % SOA_SZ)

// Typedefs
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef ${pyfr.npdtype_to_ctype(ixdtype)} ixdtype_t;

// Atomic helpers
__device__ void atomic_min_fpdtype(fpdtype_t* addr, fpdtype_t val)
{
% if pyfr.npdtype_to_ctype(fpdtype) == 'float':
    if (!signbit(val))
        atomicMin((int*) addr, __float_as_int(val));
    else
        atomicMax((unsigned int*) addr, __float_as_uint(val));
% else:
    if (!signbit(val))
        atomicMin((long long*) addr, __double_as_longlong(val));
    else
        atomicMax((unsigned long long*) addr, (unsigned long long) __double_as_longlong(val));
% endif
}

${next.body()}
