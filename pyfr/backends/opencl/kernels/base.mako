# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// AoSoA macros
#define SOA_SZ ${soasz}
#define SOA_IX(a, v, nv) ((((a) / SOA_SZ)*(nv) + (v))*SOA_SZ + (a) % SOA_SZ)

// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;

// Atomic helpers
% if pyfr.npdtype_to_ctype(fpdtype) == 'float':
#define atomic_min_pos(addr, val) atomic_min((__global int *) addr, as_int(val))
% else:
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#define atomic_min_pos(addr, val) atom_min((__global long *) addr, as_long(val))
% endif

${next.body()}
