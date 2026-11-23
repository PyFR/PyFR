<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
// Typedefs
typedef short int16_t;
typedef ushort uint16_t;
typedef unsigned int uint32_t;
typedef long int64_t;
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef ${pyfr.npdtype_to_ctype(ixdtype)} ixdtype_t;
// Unsigned integer matching fpdtype_t's width (for bit tricks)
% if pyfr.npdtype_to_ctype(fpdtype) == 'double':
typedef ulong fp_uint_t;
% else:
typedef uint fp_uint_t;
% endif
typedef ushort bf16;

static inline bf16 f32_to_bf16(float f)
{
    uint u = as_uint(f);
    return (bf16) ((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

static inline float bf16_to_f32(bf16 b)
{
    return as_float((uint) b << 16);
}

// Atomic helpers
% if pyfr.npdtype_to_ctype(fpdtype) == 'double':
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
<% itype, ftype, atomic = 'long', 'double', 'atom' %>
% else:
<% itype, ftype, atomic = 'int', 'float', 'atomic' %>
% endif
% for aspace in ['__global', '__local']:
% for op, op_pos, op_neg in [('min', 'min', 'max'), ('max', 'max', 'min')]:
__attribute__((overloadable))
void atomic_${op}_fpdtype(${aspace} fpdtype_t *addr, fpdtype_t val)
{
    if (!signbit(val))
        ${atomic}_${op_pos}((volatile ${aspace} ${itype} *) addr, as_${itype}(val));
    else
        ${atomic}_${op_neg}((volatile ${aspace} u${itype} *) addr, as_u${itype}(val));
}
% endfor
__attribute__((overloadable))
void atomic_sum_fpdtype(${aspace} fpdtype_t *addr, fpdtype_t val)
{
    u${itype} o = as_u${itype}(*addr), e;

    do
    {
        e = o;
        o = ${atomic}_cmpxchg((volatile ${aspace} u${itype} *) addr, e, as_u${itype}(as_${ftype}(e) + val));
    } while (o != e);
}
% endfor

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Vector loads
#define PYFR_LOAD_VEC(dst, src) \
    vstore4(vload4(0, (__global const uint*) (src)), 0, (uint*) (dst))

#define PYFR_BLOCK_ID get_group_id(0)
#define PYFR_BLOCK_ID_Y get_group_id(1)
#define PYFR_SYNC_GMEM_THREADS() work_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)

#define TILED_IX(e, r, c, block_sz, tile_sz) \
    ((e) * (block_sz) * (block_sz) + \
     ((r) / (tile_sz)) * ((block_sz) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((c) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((r) % (tile_sz)) * (tile_sz) + \
     ((c) % (tile_sz)))

// Global thread position
#define PYFR_GLOBAL_ID_X get_global_id(0)
#define PYFR_GLOBAL_ID_Y get_global_id(1)

${next.body()}
