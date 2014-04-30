# -*- coding: utf-8 -*-

from math import sqrt

import numpy as np

from pyfr.bases.base import BaseBasis


class TriBasis(BaseBasis):
    name = 'tri'
    ndims = 2

    # nspts = n*(n + 1)/2
    nspts_coeffs = [1, 1, 0]
    nspts_cdenom = 2

    # Faces: type, reference-to-face projection, normal, relative area
    faces = [
        ('line', lambda s: (s, -1), (0, -1), 1),
        ('line', lambda s: (-s, s), (1/sqrt(2), 1/sqrt(2)), sqrt(2)),
        ('line', lambda s: (-1, s), (-1, 0), 1),
    ]

    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)

        return [(p, q)
                for i, q in enumerate(pts1d)
                for p in pts1d[:(sptord + 1 - i)]]

    @property
    def nupts(self):
        return (self.order + 1)*(self.order + 2) // 2


class TetBasis(BaseBasis):
    name = 'tet'
    ndims = 3

    # nspts = n*(n + 1)*(n + 2)/6
    nspts_coeffs = [1, 3, 2, 0]
    nspts_cdenom = 6

    # Faces: type, reference-to-face projection, normal, relative area
    faces = [
        ('tri', lambda s, t: (s, t, -1), (0, 0, -1), 1),
        ('tri', lambda s, t: (s, -1, t), (0, -1, 0), 1),
        ('tri', lambda s, t: (-1, t, s), (-1, 0, 0), 1),
        ('tri', lambda s, t: (s, t, -s - t - 1),
         (1/sqrt(3), 1/sqrt(3), 1/sqrt(3)), sqrt(3)),
    ]

    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)

        return [(p, q, r)
                for i, r in enumerate(pts1d)
                for j, q in enumerate(pts1d[:(sptord + 1 - i)])
                for p in pts1d[:(sptord + 1 - i - j)]]

    @property
    def nupts(self):
        return (self.order + 1)*(self.order + 2)*(self.order + 3) // 6
