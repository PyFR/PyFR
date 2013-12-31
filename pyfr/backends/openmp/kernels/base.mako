# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <omp.h>
#include <stdlib.h>
#include <tgmath.h>

#define PYFR_ALIGN_BYTES 32
#define PYFR_NOINLINE __attribute__ ((noinline))

#ifdef __ICC
# define PYFR_ALIGNED(x) __assume_aligned(x, PYFR_ALIGN_BYTES)
#else
# define PYFR_ALIGNED(x) x = __builtin_assume_aligned(x, PYFR_ALIGN_BYTES)
#endif

#define PYFR_MIN(a, b) ((a) < (b) ? (a) : (b))

// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;

// OpenMP static loop scheduling functions
<%include file='loop-sched'/>

${next.body()}
