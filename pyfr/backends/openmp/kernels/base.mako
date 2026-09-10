<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <omp.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdlib.h>
#include <tgmath.h>

#define SOA_SZ ${soasz}
#define BLK_SZ ${csubsz}

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef ${pyfr.npdtype_to_ctype(ixdtype)} ixdtype_t;
typedef _Float16 half;
typedef unsigned short bf16;

static inline bf16 f32_to_bf16(float f)
{
    union { float f; unsigned int u; } v = { .f = f };
    return (bf16) ((v.u + 0x7FFF + ((v.u >> 16) & 1)) >> 16);
}

static inline float bf16_to_f32(bf16 b)
{
    union { unsigned int u; float f; } v = { .u = (unsigned int) b << 16 };
    return v.f;
}

// Atomic helpers
#define atomic_min_fpdtype(addr, val) _Pragma("omp atomic compare") if ((val) < *(addr)) { *(addr) = (val); }
#define atomic_max_fpdtype(addr, val) _Pragma("omp atomic compare") if ((val) > *(addr)) { *(addr) = (val); }


// Thread/block helpers (no-op for OpenMP)
#define PYFR_THREAD_ID 0
#define PYFR_BLOCK_ID 0
#define PYFR_BLOCK_ID_Y 0
#define PYFR_SYNC_THREADS()
#define PYFR_SHARED
#define PYFR_GMEM
#define PYFR_LMEM

// Tiled layout indexing for block-diagonal matrices
#define TILED_IX(e, r, c, block_sz, tile_sz) \
    ((e) * (block_sz) * (block_sz) + \
     ((r) / (tile_sz)) * ((block_sz) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((c) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((r) % (tile_sz)) * (tile_sz) + \
     ((c) % (tile_sz)))

// FP-precise block support
#define PYFR_FP_PRECISE_BEGIN

${next.body()}
