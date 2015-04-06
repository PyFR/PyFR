# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class BasePlugin(object, metaclass=ABCMeta):
    name = None

    @abstractmethod
    def __init__(self, intg, cfgsect):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect

        self.ndims = intg.system.ndims
        self.nvars = intg.system.nvars

    @abstractmethod
    def __call__(self, intg):
        pass
