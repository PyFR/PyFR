# -*- coding: utf-8 -*-
<%inherit file='base'/>

#include <mkl.h>

void
gemm(long *M, long *N, long *K,
     fpdtype_t **A, fpdtype_t **B, fpdtype_t **C,
     long *lda, long *ldb, long *ldc,
     double *alpha, double *beta)
{
    ${cblas_gemm}(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            *M, *N, *K, *alpha, *A, *lda, *B, *ldb, *beta, *C, *ldc);
}
