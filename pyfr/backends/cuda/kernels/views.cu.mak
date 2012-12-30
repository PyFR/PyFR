# -*- coding: utf-8 -*-

#ifndef _PYFR_VIEWS
#define _PYFR_VIEWS

#define READ_VIEW(dst, src_v, src_vstri, idx, nvec) \
    for (int _i = 0; _i < nvec; ++_i)               \
        dst[_i] = src_v[idx][src_vstri[idx]*_i]

#define READ_VIEW_V(dst, src_v, src_vstri, idx, ncol, nrow, nvec) \
    for (int _j = 0; _j < nrow; ++_j)                             \
        READ_VIEW(dst[_j], src_v, src_vstri, _j*ncol + idx, nvec)

#define WRITE_VIEW(dst_v, dst_vstri, src, idx, nvec) \
    for (int _i = 0; _i < nvec; ++_i)                \
        dst_v[idx][dst_vstri[idx]*_i] = src[_i]

#define READ_MPIM(dst, src_m, idx, ncol, nvec) \
    for (int _i = 0; _i < nvec; ++_i)          \
        dst[_i] = src_m[ncol*_i + idx]

#define READ_MPIM_V(dst, src_m, idx, ncol, nrow, nvec)       \
    for (int _i = 0; _i < nrow; ++_i)                        \
        for (int _j = 0; _j < nvec; ++_j)                     \
            dst[_j][_j] = src_m[_i*ncol*nvec + ncol*_j + idx]

#endif // _PYFR_VIEWS
