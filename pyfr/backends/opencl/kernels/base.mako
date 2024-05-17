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
% if pyfr.npdtype_to_ctype(fpdtype) == 'float':
void atomic_min_fpdtype(__global const fpdtype_t *addr, fpdtype_t val)
{
    if (!signbit(val))
        atomic_min((__global int *) addr, as_int(val));
    else
        atomic_max((__global uint *) addr, as_uint(val));
}
% else:
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
void atomic_min_fpdtype(__global const fpdtype_t *addr, fpdtype_t val)
{
    if (!signbit(val))
        atom_min((__global long *) addr, as_long(val));
    else
        atom_max((__global ulong *) addr, as_ulong(val));
}
% endif

${next.body()}
