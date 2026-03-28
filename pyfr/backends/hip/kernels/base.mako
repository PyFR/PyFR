<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// AoSoA macros
#define SOA_SZ ${soasz}
#define SOA_IX(a, v, nv) ((((a) / SOA_SZ)*(nv) + (v))*SOA_SZ + (a) % SOA_SZ)

// Typedefs
#if defined(__HIPCC__) && (HIP_VERSION_MAJOR >= 7)
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef ${pyfr.npdtype_to_ctype(ixdtype)} ixdtype_t;

// Atomic helpers
% for op, op_pos, op_neg in [('min', 'Min', 'Max'), ('max', 'Max', 'Min')]:
__device__ void atomic_${op}_fpdtype(fpdtype_t* addr, fpdtype_t val)
{
% if pyfr.npdtype_to_ctype(fpdtype) == 'float':
    if (!signbit(val))
        atomic${op_pos}((int*) addr, __float_as_int(val));
    else
        atomic${op_neg}((unsigned int*) addr, __float_as_uint(val));
% else:
    if (!signbit(val))
        atomic${op_pos}((long long*) addr, __double_as_longlong(val));
    else
        atomic${op_neg}((unsigned long long*) addr, (unsigned long long) __double_as_longlong(val));
% endif
}
% endfor

__device__ void atomic_sum_fpdtype(fpdtype_t* addr, fpdtype_t val)
{
    atomicAdd(addr, val);
}

// FP-precise block support
#define PYFR_FP_PRECISE_BEGIN _Pragma("clang fp reassociate(off) contract(off)")

${next.body()}
