# -*- coding: utf-8 -*-

<%include file='common.h.mak' />

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
        size_t offN, tN;
        static_omp_sched(N, &offN, &tN);

        gemm(ROW_MAJOR, NO_TRANS, NO_TRANS, M, tN, K,
             alpha, A, lda, B + offN, ldb, beta, C + offN, ldc);
    }
}
