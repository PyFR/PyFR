<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <metal_stdlib>

using namespace metal;
// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef ${pyfr.npdtype_to_ctype(ixdtype)} ixdtype_t;
// Unsigned integer matching fpdtype_t's width (for bit tricks)
% if pyfr.npdtype_to_ctype(fpdtype) == 'double':
typedef ulong fp_uint_t;
% else:
typedef uint fp_uint_t;
% endif
typedef bfloat bf16;

inline bf16 f32_to_bf16(float f) { return (bf16) f; }
inline float bf16_to_f32(bf16 b) { return (float) b; }

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


// Vector loads
#define PYFR_LOAD_VEC(dst, src) \
    do \
    { \
        uint4 _v = *(device const uint4*) (src); \
        for (int _i = 0; _i < 16; _i++) \
            ((thread uchar*) (dst))[_i] = ((thread const uchar*) &_v)[_i]; \
    } while (0)

// Thread/block helpers
#define PYFR_BLOCK_ID bid_.x
#define PYFR_BLOCK_ID_Y bid_.y
#define PYFR_SYNC_GMEM_THREADS() threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)

// Tiled layout indexing for block-diagonal matrices
#define TILED_IX(e, r, c, block_sz, tile_sz) \
    ((e) * (block_sz) * (block_sz) + \
     ((r) / (tile_sz)) * ((block_sz) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((c) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((r) % (tile_sz)) * (tile_sz) + \
     ((c) % (tile_sz)))

${next.body()}
