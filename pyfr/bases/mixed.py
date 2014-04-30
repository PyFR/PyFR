# -*- coding: utf-8 -*-

from math import sqrt

import numpy as np

from pyfr.bases.base import BaseBasis


class PriBasis(BaseBasis):
    name = 'pri'
    ndims = 3

    # nspts = n^2*(n + 1)/2
    nspts_coeffs = [1, 1, 0, 0]
    nspts_cdenom = 2

    # Faces: type, reference-to-face projection, normal, relative area
    faces = [
        ('tri', lambda s, t: (s, t, -1), (0, 0, -1), 1),
        ('tri', lambda s, t: (s, t, 1), (0, 0, 1), 1),
        ('quad', lambda s, t: (s, -1, t), (0, -1, 0), 1),
        ('quad', lambda s, t: (-s, s, t), (1/sqrt(2), 1/sqrt(2), 0), sqrt(2)),
        ('quad', lambda s, t: (-1, s, t), (-1, 0, 0), 1),
    ]

    @classmethod
    def std_ele(cls, sptord):
        pts1d = np.linspace(-1, 1, sptord + 1)

        return [(p, q, r)
                for r in pts1d
                for i, q in enumerate(pts1d)
                for p in pts1d[:(sptord + 1 - i)]]

    @property
    def nupts(self):
        return (self.order + 1)**2*(self.order + 2) // 2
