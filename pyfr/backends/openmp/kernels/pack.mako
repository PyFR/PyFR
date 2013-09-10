# -*- coding: utf-8 -*-

<%include file='common' />

void
pack_view(int nrow, int ncol,
          ${dtype} **vptr, int *vstri, ${dtype} *pmat,
          int ldp, int lds, int ldm)
{
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < ncol; j++)
        {
            ${dtype} *ptr = vptr[i*ldp + j];
            int stride = vstri[i*lds + j];

        % for k in range(vlen):
            pmat[i*ldm + ${k}*ncol + j] = ptr[${k}*stride];
        % endfor
        }
    }
}
