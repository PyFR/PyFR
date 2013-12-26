# -*- coding: utf-8 -*-

void
pack_view(int nrow,
          int ncol,
          const ${dtype} *__restrict__ v,
          const int *__restrict__ vix,
          const int *__restrict__ vstri,
          ${dtype} *__restrict__ pmat)
{
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < ncol; j++)
        {
            const ${dtype} *ptr = v + vix[i*ncol + j];
            int stride = vstri[i*ncol + j];

        % for k in range(vlen):
            pmat[i*ncol*${vlen} + ${k}*ncol + j] = ptr[${k}*stride];
        % endfor
        }
    }
}
