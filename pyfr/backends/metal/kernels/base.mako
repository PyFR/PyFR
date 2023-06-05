<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <metal_stdlib>

using namespace metal;

// AoSoA macros
#define SOA_SZ ${soasz}
#define SOA_IX(a, v, nv) ((((a) / SOA_SZ)*(nv) + (v))*SOA_SZ + (a) % SOA_SZ)

// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;

// Atomic helpers
inline void atomic_min_fpdtype(device fpdtype_t* addr, fpdtype_t val)
{
    union { float f; int i; uint u; } u; u.f = val;
    if (!signbit(val))
        atomic_fetch_min_explicit((device atomic_int*) addr,
                                  u.i, memory_order_relaxed);
    else
        atomic_fetch_max_explicit((device atomic_uint*) addr,
                                  u.u, memory_order_relaxed);
}

${next.body()}
