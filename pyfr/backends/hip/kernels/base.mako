<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// AoSoA macros
#define SOA_SZ ${soasz}
#define SOA_IX(a, v, nv) ((((a) / SOA_SZ)*(nv) + (v))*SOA_SZ + (a) % SOA_SZ)

// Typedefs
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef ${pyfr.npdtype_to_ctype(ixdtype)} ixdtype_t;

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

#define PYFR_THREAD_ID threadIdx.x
#define PYFR_BLOCK_ID blockIdx.x
#define PYFR_BLOCK_ID_Y blockIdx.y
#define PYFR_SYNC_THREADS() __syncthreads()
#define PYFR_SYNC_GMEM_THREADS() do { __threadfence_block(); __syncthreads(); } while (0)
#define PYFR_SHARED __shared__
#define PYFR_GMEM
#define PYFR_LMEM

#define TILED_IX(e, r, c, block_sz, tile_sz) \
    ((e) * (block_sz) * (block_sz) + \
     ((r) / (tile_sz)) * ((block_sz) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((c) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((r) % (tile_sz)) * (tile_sz) + \
     ((c) % (tile_sz)))

// Global thread position
#define PYFR_GLOBAL_ID_X (ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x)
#define PYFR_GLOBAL_ID_Y (blockIdx.y*blockDim.y + threadIdx.y)

// Subgroup shuffle primitive
#define PYFR_SHFL_XOR(v, o) __shfl_xor(v, o)

<%def name="argmax_storage(ftype)">
<% nsg = -(-nthreads // sgsize) %>
    PYFR_SHARED ${ftype} argmaxv[${nsg}];
    PYFR_SHARED short argmaxi[${nsg}];
</%def>

<%def name="argmax_reduce(ftype, val, idx, dst)">
<% nsg = -(-nthreads // sgsize) %>
    {
        short amsg = PYFR_THREAD_ID / ${sgsize}, amsl = PYFR_THREAD_ID % ${sgsize};
        ${ftype} amv = ${val}; int ami = ${idx};
        for (int amo = ${sgsize} / 2; amo > 0; amo >>= 1) {
            ${ftype} amov = PYFR_SHFL_XOR(amv, amo);
            int amoi = PYFR_SHFL_XOR(ami, amo);
            if (amov > amv || (amov == amv && amoi < ami)) { amv = amov; ami = amoi; }
        }
        if (amsl == 0) { argmaxv[amsg] = amv; argmaxi[amsg] = ami; }
        PYFR_SYNC_THREADS();
        amv = argmaxv[0]; ami = argmaxi[0];
        for (int amw = 1; amw < ${nsg}; amw++)
            if (argmaxv[amw] > amv || (argmaxv[amw] == amv && argmaxi[amw] < ami)) { amv = argmaxv[amw]; ami = argmaxi[amw]; }
        if (PYFR_THREAD_ID == 0) ${dst} = ami;
        PYFR_SYNC_THREADS();
    }
</%def>

// FP-precise block support
#define PYFR_FP_PRECISE_BEGIN _Pragma("clang fp reassociate(off) contract(off)")

<%def name="_kdecl(name, bounds)">\
% if bounds:
__global__ __launch_bounds__(${bounds}) void ${name}\
% else:
__global__ void ${name}\
% endif
</%def>
<%def name="_karg(intent, t, n)">\
% if intent == 'in':
const ${t}* __restrict__ ${n}\
% elif intent == 'out':
${t}* __restrict__ ${n}\
% else:
${t} ${n}\
% endif
</%def>

${next.body()}
