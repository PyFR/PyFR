<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// AoSoA macros
#define SOA_SZ ${soasz}
#define SOA_IX(a, v, nv) ((((a) / SOA_SZ)*(nv) + (v))*SOA_SZ + (a) % SOA_SZ)

// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;

// Atomic helpers
% if pyfr.npdtype_to_ctype(fpdtype) == 'float':
#define atomic_min_pos(addr, val) atomicMin((int*) addr, __float_as_int(val))
% else:
#define atomic_min_pos(addr, val) atomicMin((long long*) addr, __double_as_longlong(val))
% endif

${next.body()}
