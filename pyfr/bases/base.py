# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

class BasisBase(object):
    name = None
    ndims = -1

    def __init__(self, dims, cfg):
        self._dims = dims
        self._cfg = cfg
        self._order = int(cfg.get('scheme', 'order'))

        if self.ndims != len(dims):
            raise ValueError('Invalid dimension symbols')

    @property
    def dims(self):
        return self._dims

    @abstractmethod
    def gen_upts(self):
        pass

    @abstractmethod
    def gen_fpts(self):
        pass

    @abstractmethod
    def gen_spts(self):
        pass

    @abstractmethod
    def get_nfpts_for_face(self, face):
        pass

    @abstractmethod
    def get_fpts_for_face(self, face, rtag):
        pass
