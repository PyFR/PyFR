# -*- coding: utf-8 -*-

from pyfr.backends.base import Backend as BaseBackend
from pyfr.backends.cuda import CudaBackend
from pyfr.util import subclass_map


def get_backend(name, cfg):
    backend_map = subclass_map(BaseBackend, 'name')

    return backend_map[name.lower()](cfg)
