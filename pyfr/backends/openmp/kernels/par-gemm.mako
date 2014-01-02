# -*- coding: utf-8 -*-
<%inherit file='base'/>

// CBLAS GEMM constants
#define ROW_MAJOR 101
#define NO_TRANS  111

// CBLAS GEMM prototype
typedef void (*cblas_gemm_t)(int, int, int,
                             int, int, int,
                             fpdtype_t, const fpdtype_t *, int,
                             const fpdtype_t *, int,
                             fpdtype_t, fpdtype_t *, int);

void
par_gemm(cblas_gemm_t gemm, int M, int N, int K,
         fpdtype_t alpha, const fpdtype_t *A, int lda,
         const fpdtype_t *B, int ldb,
         fpdtype_t beta, fpdtype_t *C, int ldc)
{
    #pragma omp parallel
    {
        int begin, end;
        loop_sched_1d(N, PYFR_ALIGN_BYTES / sizeof(fpdtype_t), &begin, &end);

        gemm(ROW_MAJOR, NO_TRANS, NO_TRANS, M, end - begin, K,
             alpha, A, lda, B + begin, ldb, beta, C + begin, ldc);
    }
}
