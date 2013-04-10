# -*- coding: utf-8 -*-

#ifndef _PYFR_COMMON_H
#define _PYFR_COMMON_H

#include <stdlib.h>
#include <tgmath.h>
#include <omp.h>

#define ALIGN_BYTES 32
#define ASSUME_ALIGNED(x) x = __builtin_assume_aligned(x, ALIGN_BYTES)

#define NOINLINE __attribute__ ((noinline))


/**
 * Performs static OpenMP scheduling for a #parallel block such that work is
 * distributed between all of the currently active threads.
 */
static inline void
static_omp_sched(size_t N, size_t *tstart, size_t *tn)
{
    int tid = omp_get_thread_num();
    int cnt = omp_get_num_threads();

    // More items than threads
    if (cnt < N)
    {
        int tc = N / cnt;

        *tstart = tid*tc;
        *tn = (tid < cnt - 1) ? tc : tc + N % cnt;
    }
    // Less items than threads; only use the first N threads
    else if (tid < N)
    {
        *tstart = tid;
        *tn = 1;
    }
    // Less items than threads; our tid > N
    else
    {
        *tstart = 0;
        *tn = 0;
    }
}

#endif // _PYFR_COMMON_H
