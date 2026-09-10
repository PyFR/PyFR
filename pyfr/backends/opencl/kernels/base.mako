<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// AoSoA macros
#define SOA_SZ ${soasz}
#define SOA_IX(a, v, nv) ((((a) / SOA_SZ)*(nv) + (v))*SOA_SZ + (a) % SOA_SZ)

// Typedefs
typedef unsigned int uint32_t;
typedef long int64_t;
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef ${pyfr.npdtype_to_ctype(ixdtype)} ixdtype_t;
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

#define PYFR_THREAD_ID get_local_id(0)
#define PYFR_BLOCK_ID get_group_id(0)
#define PYFR_BLOCK_ID_Y get_group_id(1)
#define PYFR_SYNC_THREADS() work_group_barrier(CLK_LOCAL_MEM_FENCE)
#define PYFR_SYNC_GMEM_THREADS() work_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
#define PYFR_SHARED __local
#define PYFR_GMEM __global
#define PYFR_LMEM __local

#define TILED_IX(e, r, c, block_sz, tile_sz) \
    ((e) * (block_sz) * (block_sz) + \
     ((r) / (tile_sz)) * ((block_sz) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((c) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((r) % (tile_sz)) * (tile_sz) + \
     ((c) % (tile_sz)))

// Global thread position
#define PYFR_GLOBAL_ID_X get_global_id(0)
#define PYFR_GLOBAL_ID_Y get_global_id(1)

% if wg_reduce_compat:
// OpenCL 2.0 work-group reductions are unavailable on some runtimes
// (e.g. PoCL 1.8); emulate them through shared-memory tree reductions.
// Declared as macros so the __local scratch array lives in the kernel
// function itself, as required by OpenCL C.
#define PYFR_WG_REDUCE_DECLARE(t, name, expr)                           \
    __local t pyfr_wg_##name##_smem[PYFR_WG_LSIZE_MAX];                 \
    __attribute__((overloadable))                                       \
    t work_group_reduce_##name(t v)                                     \
    {                                                                   \
        int lid = (int)get_local_id(0), lsz = (int)get_local_size(0);   \
        smem_next: ;                                                    \
        pyfr_wg_##name##_smem[lid] = v;                                 \
        work_group_barrier(CLK_LOCAL_MEM_FENCE);                        \
        for (int off = 1; off < lsz; off <<= 1)                         \
        {                                                               \
            if (lid % (off << 1) == 0 && lid + off < lsz)               \
                pyfr_wg_##name##_smem[lid] = expr;                      \
            work_group_barrier(CLK_LOCAL_MEM_FENCE);                    \
        }                                                               \
        return pyfr_wg_##name##_smem[0];                                \
    }

#define PYFR_WG_LSIZE_MAX 1024

<%
    wg_reduces = [('double', 'add', 'pyfr_wg_add_smem[lid] + pyfr_wg_add_smem[lid + off]'),
                  ('double', 'min', 'min(pyfr_wg_min_smem[lid], pyfr_wg_min_smem[lid + off])'),
                  ('double', 'max', 'max(pyfr_wg_max_smem[lid], pyfr_wg_max_smem[lid + off])'),
                  ('float', 'add', 'pyfr_wg_add_smem[lid] + pyfr_wg_add_smem[lid + off]'),
                  ('float', 'min', 'min(pyfr_wg_min_smem[lid], pyfr_wg_min_smem[lid + off])'),
                  ('float', 'max', 'max(pyfr_wg_max_smem[lid], pyfr_wg_max_smem[lid + off])'),
                  ('int', 'min', 'min(pyfr_wg_min_smem[lid], pyfr_wg_min_smem[lid + off])'),
                  ('int', 'max', 'max(pyfr_wg_max_smem[lid], pyfr_wg_max_smem[lid + off])')]
%>
#define PYFR_WG_REDUCE_IMPL()                                           \
do                                                                      \
{                                                                       \
% for t, name, expr in wg_reduces:
    PYFR_WG_REDUCE_DECLARE(${t}, ${name}, ${expr})                      \
% endfor
} while (0)
% endif

<%def name="argmax_storage(ftype)"></%def>

<%def name="argmax_reduce(ftype, val, idx, dst)">
    {
        ${ftype} amv = work_group_reduce_max(${val});
        int ami = work_group_reduce_min((${val} == amv) ? ${idx} : 0x7fffffff);
        if (PYFR_THREAD_ID == 0) ${dst} = ami;
        PYFR_SYNC_THREADS();
    }
</%def>

// FP-precise block support
#ifdef __clang__
#define PYFR_FP_PRECISE_BEGIN _Pragma("clang fp reassociate(off) contract(off)")
#else
#define PYFR_FP_PRECISE_BEGIN
#endif

<%def name="_kdecl(name, bounds)">\
% if bounds:
__kernel __attribute__((reqd_work_group_size(${bounds}, 1, 1))) void ${name}\
% else:
__kernel void ${name}\
% endif
</%def>
<%def name="_karg(intent, t, n)">\
% if intent == 'in':
__global const ${t}* restrict ${n}\
% elif intent == 'out':
__global ${t}* restrict ${n}\
% else:
${t} ${n}\
% endif
</%def>

${next.body()}
