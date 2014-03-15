# -*- coding: utf-8 -*-

from pyfr.backends.base import BaseBackend
from pyfr.backends.cuda import CUDABackend
from pyfr.backends.opencl import OpenCLBackend
from pyfr.backends.openmp import OpenMPBackend
from pyfr.util import subclass_where


def get_backend(name, cfg):
    return subclass_where(BaseBackend, name=name.lower())(cfg)
