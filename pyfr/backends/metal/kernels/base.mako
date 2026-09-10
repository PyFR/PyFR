<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <metal_stdlib>

using namespace metal;

// AoSoA macros
#define SOA_SZ ${soasz}
#define SOA_IX(a, v, nv) ((((a) / SOA_SZ)*(nv) + (v))*SOA_SZ + (a) % SOA_SZ)

// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef ${pyfr.npdtype_to_ctype(ixdtype)} ixdtype_t;
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


// Thread/block helpers
#define PYFR_THREAD_ID tid_.x
#define PYFR_BLOCK_ID bid_.x
#define PYFR_BLOCK_ID_Y bid_.y
#define PYFR_SYNC_THREADS() threadgroup_barrier(mem_flags::mem_threadgroup)
#define PYFR_SYNC_GMEM_THREADS() threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#define PYFR_SHARED threadgroup
#define PYFR_GMEM device
#define PYFR_LMEM threadgroup

// Tiled layout indexing for block-diagonal matrices
#define TILED_IX(e, r, c, block_sz, tile_sz) \
    ((e) * (block_sz) * (block_sz) + \
     ((r) / (tile_sz)) * ((block_sz) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((c) / (tile_sz)) * (tile_sz) * (tile_sz) + \
     ((r) % (tile_sz)) * (tile_sz) + \
     ((c) % (tile_sz)))

<%def name="argmax_storage(ftype)">
<% nsg = -(-nthreads // sgsize) %>
    PYFR_SHARED ${ftype} argmaxv[${nsg}];
    PYFR_SHARED short argmaxi[${nsg}];
</%def>

<%def name="argmax_reduce(ftype, val, idx, dst)">
<% nsg = -(-nthreads // sgsize) %>
    {
        short amsg = PYFR_THREAD_ID / ${sgsize}, amsl = PYFR_THREAD_ID % ${sgsize};
        ${ftype} amv = simd_max(${val});
        int ami = simd_min((${val} == amv) ? ${idx} : 0x7fffffff);
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

<%def name="_kdecl(name, bounds)">kernel void ${name}</%def>
<%def name="_karg(intent, t, n)">\
% if intent == 'in':
device const ${t}* ${n}\
% elif intent == 'out':
device ${t}* ${n}\
% else:
constant ${t}& ${n}\
% endif
</%def>
<%def name="_kextra(body)"><%
    extra = []
    if 'PYFR_THREAD_ID' in body:
        extra.append('uint3 tid_ [[thread_position_in_threadgroup]]')
        extra.append('uint3 bid_ [[threadgroup_position_in_grid]]')
    if 'PYFR_GLOBAL_ID' in body:
        extra.append('uint2 ji [[thread_position_in_grid]]')
%>${', '.join(extra)}</%def>

${next.body()}
