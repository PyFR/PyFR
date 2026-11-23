<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
// Typedefs
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef ${pyfr.npdtype_to_ctype(ixdtype)} ixdtype_t;
// Unsigned integer matching fpdtype_t's width (for bit tricks)
% if pyfr.npdtype_to_ctype(fpdtype) == 'double':
typedef uint64_t fp_uint_t;
% else:
typedef uint32_t fp_uint_t;
% endif

// Atomic helpers
#define atomic_min_fpdtype(addr, val) atomicMin(addr, val)
#define atomic_max_fpdtype(addr, val) atomicMax(addr, val)

__device__ void atomic_sum_fpdtype(fpdtype_t* addr, fpdtype_t val)
{
    atomicAdd(addr, val);
}

struct bf16 {
    unsigned short bits;
    __device__ bf16() {}
    __device__ bf16(fpdtype_t d) {
        unsigned int u = __float_as_uint((float) d);
        bits = (unsigned short) ((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
    }
    __device__ operator fpdtype_t() const {
        return (fpdtype_t) __uint_as_float((unsigned int) bits << 16);
    }
};

__device__ static inline bf16 f32_to_bf16(float f) { return bf16(f); }
__device__ static inline float bf16_to_f32(bf16 b) { return (float) b; }

// Vector loads, streamed when the source is const
typedef unsigned int pyfr_u32x4_t __attribute__((ext_vector_type(4)));
__device__ static inline void pyfr_load_vec(void *d, const void *s)
{
    pyfr_u32x4_t v = __builtin_nontemporal_load((const pyfr_u32x4_t*) s);
    __builtin_memcpy(d, &v, 16);
}
__device__ static inline void pyfr_load_vec(void *d, void *s)
{
    pyfr_u32x4_t v = *(const pyfr_u32x4_t*) s;
    __builtin_memcpy(d, &v, 16);
}
#define PYFR_LOAD_VEC(dst, src) pyfr_load_vec(dst, src)

% if ndies > 1:
// Give each die a contiguous range of blocks
#define PYFR_BLOCK_ID ((blockIdx.x % ${ndies})*(gridDim.x / ${ndies}) + min(blockIdx.x % ${ndies}, gridDim.x % ${ndies}) + blockIdx.x / ${ndies})
% else:
#define PYFR_BLOCK_ID blockIdx.x
% endif
#define PYFR_BLOCK_ID_Y blockIdx.y
#define PYFR_SYNC_GMEM_THREADS() do { __threadfence_block(); __syncthreads(); } while (0)

#define TILED_IX(e, r, c, block_sz, tile_sz) \
    ((e) * (block_sz) * (block_sz) + \
     ((r) / (tile_sz)) * ((block_sz) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((c) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((r) % (tile_sz)) * (tile_sz) + \
     ((c) % (tile_sz)))

// Global thread position
#define PYFR_GLOBAL_ID_X (ixdtype_t(PYFR_BLOCK_ID)*blockDim.x + threadIdx.x)
#define PYFR_GLOBAL_ID_Y (blockIdx.y*blockDim.y + threadIdx.y)

${next.body()}
