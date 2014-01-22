# -*- coding: utf-8 -*-

from pyfr.backends.base import BaseBackend
from pyfr.backends.cuda import CUDABackend
from pyfr.backends.opencl import OpenCLBackend
from pyfr.backends.openmp import OpenMPBackend
from pyfr.util import subclass_map


def get_backend(name, cfg):
    backend_map = subclass_map(BaseBackend, 'name')

    return backend_map[name.lower()](cfg)
