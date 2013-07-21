# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from ctypes import CDLL
import itertools as it
import os
import subprocess
import tempfile

from pyfr.ctypesutil import platform_libname
from pyfr.nputil import npdtype_to_ctypestype
from pyfr.util import chdir, memoize, rm


class SourceModule(object):
    __metaclass__ = ABCMeta

    _dir_seq = it.count()

    def __init__(self, src, cfg):
        self._src = src
        self._cfg = cfg

        # Create a scratch directory
        tmpdir = tempfile.mkdtemp(prefix='pyfr-%d-' % next(self._dir_seq))

        try:
            with chdir(tmpdir):
                # Compile and link the source
                lname = self._build()

                # Load
                self._mod = CDLL(os.path.abspath(lname))
        finally:
            # Unless we're debugging delete the scratch directory
            if 'PYFR_DEBUG_OMP_KEEP_LIBS' not in os.environ:
                rm(tmpdir)

    def function(self, name, restype, argtypes):
        # Get the function
        fn = getattr(self._mod, name)
        fn.restype = npdtype_to_ctypestype(restype)
        fn.argtypes = [npdtype_to_ctypestype(a) for a in argtypes]

        return fn

    @abstractmethod
    def _build(self):
        pass


class GccSourceModule(SourceModule):
    def __init__(self, src, cfg):
        # Find GCC (or a compatible alternative)
        self._cc = cfg.getpath('backend-openmp', 'cc', 'cc', abs=False)

        # Delegate
        super(GccSourceModule, self).__init__(src, cfg)

    def _build(self):
        # File names
        cn, on, ln = 'tmp.c', 'tmp.o', platform_libname('tmp')

        # Write the source code out
        with open(cn, 'w') as f:
            f.write(self._src)

        # Compile
        cmd = [self._cc,
               '-std=c99',       # Enable C99 support
               '-Ofast',         # Optimise, incl. -ffast-math
               '-march=native',  # Use CPU-specific instructions
               '-fopenmp',       # Enable OpenMP support
               '-fPIC',          # Position-independent code for shared lib
               '-c', '-o', on, cn]
        out = subprocess.check_call(cmd, stderr=subprocess.STDOUT)

        # Link
        cmd = [self._cc,
               '-shared',   # Create a shared library
               '-fopenmp',  # Required for OpenMP
               '-o', ln, on,]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)

        return ln
