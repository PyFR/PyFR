# -*- coding: utf-8 -*-

import re
import uuid

from abc import ABCMeta, abstractmethod

import numpy as np


class BaseReader(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _to_raw_pyfrm(self):
        pass

    def _optimize(self, mesh):
        # Sort interior interfaces
        for f in filter(lambda f: re.match(r'^con_p\d+$', f), mesh):
            mesh[f] = mesh[f][:,np.argsort(mesh[f][0])]

    def to_pyfrm(self):
        mesh = self._to_raw_pyfrm()

        # Perform some simple optimizations on the mesh
        self._optimize(mesh)

        # Add metadata
        mesh['mesh_uuid'] = str(uuid.uuid4())

        return mesh
