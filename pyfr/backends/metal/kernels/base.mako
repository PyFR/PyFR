<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <metal_stdlib>

using namespace metal;

// AoSoA macros
#define SOA_SZ ${soasz}
#define SOA_IX(a, v, nv) ((((a) / SOA_SZ)*(nv) + (v))*SOA_SZ + (a) % SOA_SZ)

// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef ${pyfr.npdtype_to_ctype(ixdtype)} ixdtype_t;

// Atomic helpers
% for aspace in ['device', 'threadgroup']:
% for op, op_pos, op_neg in [('min', 'min', 'max'), ('max', 'max', 'min')]:
inline void atomic_${op}_fpdtype(${aspace} fpdtype_t* addr, fpdtype_t val)
{
    union { float f; int i; uint u; } u; u.f = val;
    if (!signbit(val))
        atomic_fetch_${op_pos}_explicit((${aspace} atomic_int*) addr,
                                        u.i, memory_order_relaxed);
    else
        atomic_fetch_${op_neg}_explicit((${aspace} atomic_uint*) addr,
                                        u.u, memory_order_relaxed);
}
% endfor
inline void atomic_sum_fpdtype(${aspace} fpdtype_t* addr, fpdtype_t val)
{
    union { float f; uint u; } e, d;
    e.u = atomic_load_explicit((${aspace} atomic_uint*) addr,
                               memory_order_relaxed);
    do {
        d.f = e.f + val;
    } while (!atomic_compare_exchange_weak_explicit(
        (${aspace} atomic_uint*) addr, &e.u, d.u,
        memory_order_relaxed, memory_order_relaxed));
}
% endfor

// FP-precise block support
#define PYFR_FP_PRECISE_BEGIN _Pragma("clang fp reassociate(off) contract(off)")

${next.body()}
