# -*- coding: utf-8 -*-
<%inherit file='base'/>

// GiMMiK kernel
${gimmik_mm}

void
par_gimmik_mm(int N, const fpdtype_t *B, int ldb, fpdtype_t *C, int ldc)
{
    #pragma omp parallel
    {
        int begin, end;
        loop_sched_1d(N, PYFR_ALIGN_BYTES / sizeof(fpdtype_t), &begin, &end);

        gimmik_mm(end - begin, B + begin, ldb, C + begin, ldc);
    }
}
