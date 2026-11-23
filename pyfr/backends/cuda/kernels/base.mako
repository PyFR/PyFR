<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
// Typedefs
typedef short int16_t;
typedef unsigned short uint16_t;
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

struct half {
    unsigned short bits;
    __device__ half() {}
    __device__ half(fpdtype_t d) {
        float f = (float) d;
        unsigned short h;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
        bits = h;
    }
    __device__ operator fpdtype_t() const {
        float f;
        asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
        return (fpdtype_t) f;
    }
};

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

// Vector loads, through the read-only cache when the source is const
__device__ static inline void pyfr_load_vec(void *d, const void *s)
{
    uint4 v = __ldg((const uint4*) s);
    memcpy(d, &v, sizeof(v));
}
__device__ static inline void pyfr_load_vec(void *d, void *s)
{
    uint4 v = *(const uint4*) s;
    memcpy(d, &v, sizeof(v));
}
#define PYFR_LOAD_VEC(dst, src) pyfr_load_vec(dst, src)

#define PYFR_BLOCK_ID blockIdx.x
#define PYFR_BLOCK_ID_Y blockIdx.y
#define PYFR_SYNC_GMEM_THREADS() __syncthreads()

#define TILED_IX(e, r, c, block_sz, tile_sz) \
    ((e) * (block_sz) * (block_sz) + \
     ((r) / (tile_sz)) * ((block_sz) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((c) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((r) % (tile_sz)) * (tile_sz) + \
     ((c) % (tile_sz)))

__device__ void atomic_sum_fpdtype(fpdtype_t* addr, fpdtype_t val)
{
    atomicAdd(addr, val);
}

// Global thread position
#define PYFR_GLOBAL_ID_X (ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x)
#define PYFR_GLOBAL_ID_Y (blockIdx.y*blockDim.y + threadIdx.y)

${next.body()}
