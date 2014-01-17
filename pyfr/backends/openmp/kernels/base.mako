# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <omp.h>
#include <stdlib.h>
#include <tgmath.h>

#define PYFR_ALIGN_BYTES ${alignb}
#define PYFR_NOINLINE __attribute__ ((noinline))

#define PYFR_MIN(a, b) ((a) < (b) ? (a) : (b))

// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;

// OpenMP static loop scheduling functions
<%include file='loop-sched'/>

${next.body()}
