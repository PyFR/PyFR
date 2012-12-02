# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

class BasisBase(object):
    __metaclass__ = ABCMeta

    name = None
    ndims = -1

    def __init__(self, dims, cfg):
        self._dims = dims
        self._cfg = cfg
        self._order = cfg.getint('mesh-elements', 'order')

        if self.ndims != len(dims):
            raise ValueError('Invalid dimension symbols')

    @property
    def dims(self):
        return self._dims

    @abstractmethod
    def get_upts(self):
        pass

    @abstractmethod
    def get_fpts(self):
        pass

    @abstractmethod
    def get_spts(self):
        pass

    @abstractmethod
    def get_nfpts(self):
        pass

    @abstractmethod
    def get_fpts_for_face(self, face, rtag):
        pass
