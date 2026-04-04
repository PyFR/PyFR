<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// AoSoA macros
#define SOA_SZ ${soasz}
#define SOA_IX(a, v, nv) ((((a) / SOA_SZ)*(nv) + (v))*SOA_SZ + (a) % SOA_SZ)

// Typedefs
typedef unsigned int uint32_t;
typedef long int64_t;
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef ${pyfr.npdtype_to_ctype(ixdtype)} ixdtype_t;

// Atomic helpers
% if pyfr.npdtype_to_ctype(fpdtype) == 'double':
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
% endif
% for aspace in ['__global', '__local']:
% for op, op_pos, op_neg in [('min', 'min', 'max'), ('max', 'max', 'min')]:
% if pyfr.npdtype_to_ctype(fpdtype) == 'float':
__attribute__((overloadable))
void atomic_${op}_fpdtype(${aspace} fpdtype_t *addr, fpdtype_t val)
{
    if (!signbit(val))
        atomic_${op_pos}((${aspace} volatile int *) addr, as_int(val));
    else
        atomic_${op_neg}((${aspace} volatile uint *) addr, as_uint(val));
}
% else:
__attribute__((overloadable))
void atomic_${op}_fpdtype(${aspace} fpdtype_t *addr, fpdtype_t val)
{
    if (!signbit(val))
        atom_${op_pos}((${aspace} volatile long *) addr, as_long(val));
    else
        atom_${op_neg}((${aspace} volatile ulong *) addr, as_ulong(val));
}
% endif
% endfor
<%
    if pyfr.npdtype_to_ctype(fpdtype) == 'float':
        itype, as_f, cmpxchg = 'uint', 'as_float', 'atomic_cmpxchg'
    else:
        itype, as_f, cmpxchg = 'ulong', 'as_double', 'atom_cmpxchg'
%>\
__attribute__((overloadable))
void atomic_sum_fpdtype(${aspace} fpdtype_t *addr, fpdtype_t val)
{
    ${itype} e = as_${itype}(*addr), o;
    do {
        o = e;
        e = ${cmpxchg}((${aspace} volatile ${itype} *) addr,
                        e, as_${itype}(${as_f}(e) + val));
    } while (e != o);
}
% endfor

// FP-precise block support
#ifdef __clang__
#define PYFR_FP_PRECISE_BEGIN _Pragma("clang fp reassociate(off) contract(off)")
#else
#define PYFR_FP_PRECISE_BEGIN
#endif

${next.body()}
