# -*- coding: utf-8 -*-
#include <omp.h>

/**
 * Performs static OpenMP scheduling for a #parallel block such that work is
 * distributed between all of the currently active threads.
 */
static inline void
static_omp_sched(int N, int *tstart, int *tn)
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

// CBLAS GEMM constants
#define ROW_MAJOR 101
#define NO_TRANS  111

// CBLAS GEMM prototype
typedef void (*cblas_gemm_t)(int, int, int,
                             int, int, int,
                             ${dtype}, const ${dtype} *, int,
                             const ${dtype} *, int,
                             ${dtype}, ${dtype} *, int);

void
par_gemm(cblas_gemm_t gemm, int M, int N, int K,
         ${dtype} alpha, const ${dtype} *A, int lda,
         const ${dtype} *B, int ldb,
         ${dtype} beta, ${dtype} *C, int ldc)
{
    #pragma omp parallel
    {
        int offN, tN;
        static_omp_sched(N, &offN, &tN);

        gemm(ROW_MAJOR, NO_TRANS, NO_TRANS, M, tN, K,
             alpha, A, lda, B + offN, ldb, beta, C + offN, ldc);
    }
}
