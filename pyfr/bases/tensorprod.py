# -*- coding: utf-8 -*-

import itertools as it

import numpy as np

from pyfr.bases.base import BaseBasis


class TensorProdBasis(object):
    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)
        return list(p[::-1] for p in it.product(pts1d, repeat=cls.ndims))

    @property
    def nupts(self):
        return (self.order + 1)**self.ndims


class QuadBasis(TensorProdBasis, BaseBasis):
    name = 'quad'
    ndims = 2

    # nspts = n^2
    nspts_coeffs = [1, 0, 0]
    nspts_cdenom = 1

    # Faces: type, reference-to-face projection, normal, relative area
    faces = [
        ('line', lambda s: (s, -1), (0, -1), 1),
        ('line', lambda s: (1, s), (1, 0), 1),
        ('line', lambda s: (s, 1), (0, 1), 1),
        ('line', lambda s: (-1, s), (-1, 0), 1),
    ]


class HexBasis(TensorProdBasis, BaseBasis):
    name = 'hex'
    ndims = 3

    # nspts = n^3
    nspts_coeffs = [1, 0, 0, 0]
    nspts_cdenom = 1

    # Faces: type, reference-to-face projection, normal, relative area
    faces = [
        ('quad', lambda s, t: (s, t, -1), (0, 0, -1), 1),
        ('quad', lambda s, t: (s, -1, t), (0, -1, 0), 1),
        ('quad', lambda s, t: (1, s, t), (1, 0, 0), 1),
        ('quad', lambda s, t: (s, 1, t), (0, 1, 0), 1),
        ('quad', lambda s, t: (-1, s, t), (-1, 0, 0), 1),
        ('quad', lambda s, t: (s, t, 1), (0, 0, 1), 1),
    ]
