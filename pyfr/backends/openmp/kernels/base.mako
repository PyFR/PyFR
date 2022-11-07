<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <omp.h>
#include <stdlib.h>
#include <tgmath.h>

#define SOA_SZ ${soasz}
#define BLK_SZ ${csubsz}

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;

// Atomic helpers
#define atomic_min_pos(addr, val) _Pragma("omp atomic compare") if ((val) < *(addr)) { *(addr) = (val); }

${next.body()}
