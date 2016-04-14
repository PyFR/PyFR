# -*- coding: utf-8 -*-

static inline int
gcd(int a, int b)
{
    return (a == 0) ? b : gcd(b % a, a);
}

static inline void
loop_sched_1d(int n, int align, int *b, int *e)
{
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();

    // Round up n to be a multiple of nth
    int rn = n + nth - 1 - (n - 1) % nth;

    // Nominal tile size
    int sz = rn / nth;

    // Handle alignment
    sz += align - 1 - (sz - 1) % align;

    // Assign the starting and ending index
    *b = sz * tid;
    *e = min(*b + sz, n);

    // Clamp
    if (*b >= n)
        *b = *e = 0;
}

static inline void
loop_sched_2d(int nrow, int ncol, int colalign,
              int *rowb, int *rowe, int *colb, int *cole)
{
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();

    // Distribute threads
    int nrowth = gcd(nrow, nth);
    int ncolth = nth / nrowth;

    // Row and column indices for our thread
    int rowix = tid / ncolth;
    int colix = tid % ncolth;

    // Round up ncol to be a multiple of ncolth
    int rncol = ncol + ncolth - 1 - (ncol - 1) % ncolth;

    // Nominal tile size
    int ntilerow = nrow / nrowth;
    int ntilecol = rncol / ncolth;

    // Handle column alignment
    ntilecol += colalign - 1 - (ntilecol - 1) % colalign;

    // Assign the starting and ending row to each thread
    *rowb = ntilerow * rowix;
    *rowe = *rowb + ntilerow;

    // Assign the starting and ending column to each thread
    *colb = ntilecol * colix;
    *cole = min(*colb + ntilecol, ncol);

    // Clamp
    if (*colb >= ncol)
        *colb = *cole = 0;
}
