# -*- coding: utf-8 -*-

<%include file='common.h.mako' />

void
pack_view(size_t nrow, size_t ncol,
          ${dtype} **vptr, int *vstri, ${dtype} *pmat,
          size_t ldp, size_t lds, size_t ldm)
{
    for (size_t i = 0; i < nrow; i++)
    {
        for (size_t j = 0; j < ncol; j++)
        {
            ${dtype} *ptr = vptr[i*ldp + j];
            int stride = vstri[i*lds + j];

        % for k in range(vlen):
            pmat[i*ldm + ${k}*ncol + j] = ptr[${k}*stride];
        % endfor
        }
    }
}
