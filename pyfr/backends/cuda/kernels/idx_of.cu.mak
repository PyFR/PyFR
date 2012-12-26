# -*- coding: utf-8 -*-

#ifndef _PYFR_IDX_OF
#define _PYFR_IDX_OF

#define IDX_OF(i, j, ldim) ((i)*(ldim) + (j))

#define U_IDX_OF(upt, ele, var, nele, ldim) \
    IDX_OF(upt, nele*var + ele, ldim)

#define GRAD_U_IDX_OF(upt, ele, dim, var, nele, ldim) \
    IDX_OF(upt*var + upt, nele*dim + ele, ldim)

#define F_IDX_OF(upt, ele, fvar, var, nupt, nele, ldim) \
    IDX_OF(nupt*fvar + upt, nele*var + ele, ldim)

% if ndims is not UNDEFINED:
#define SMAT_IDX_OF(upt, ele, row, col, nele, ldim) \
    IDX_OF(upt, nele*(${ndims}*row + col) + ele, ldim)

#define JMAT_IDX_OF SMAT_IDX_OF
% endif

#endif // _PYFR_IDX_OF
