<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <tgmath.h>

#define SOA_SZ ${soasz}
#define BLK_SZ ${csubsz}

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef ${pyfr.npdtype_to_ctype(ixdtype)} ixdtype_t;

// Atomic helpers
#define atomic_min_fpdtype(addr, val) _Pragma("omp atomic compare") if ((val) < *(addr)) { *(addr) = (val); }
#define atomic_max_fpdtype(addr, val) _Pragma("omp atomic compare") if ((val) > *(addr)) { *(addr) = (val); }
#define atomic_sum_fpdtype(addr, val) _Pragma("omp atomic") *(addr) += (val)

// FP-precise block support
#define PYFR_FP_PRECISE_BEGIN

${next.body()}
