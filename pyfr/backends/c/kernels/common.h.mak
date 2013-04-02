# -*- coding: utf-8 -*-

#ifndef _PYFR_COMMON_H
#define _PYFR_COMMON_H

#include <stdlib.h>
#include <tgmath.h>

#define ALIGN_BYTES 32
#define ASSUME_ALIGNED(x) x = __builtin_assume_aligned(x, ALIGN_BYTES)

#define NOINLINE __attribute__ ((noinline))

#endif // _PYFR_COMMON_H
