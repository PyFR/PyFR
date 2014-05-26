# -*- coding: utf-8 -*-

from ctypes import CDLL, POINTER, c_int64, c_float, c_void_p

import numpy as np

from pyfr.ctypesutil import platform_libname
from pyfr.partitioners.base import BasePartitioner


# Possible METIS exception types
METISError = type('METISError', (Exception,), {})
METISErrorInput = type('METISErrorInput', (METISError,), {})
METISErrorMemory = type('METISErrorMemory', (METISError,), {})


class METISWrappers(object):
    # Possible return codes
    _statuses = {
        -2: METISErrorInput,
        -3: METISErrorMemory,
        -4: METISError
    }

    def __init__(self):
        try:
            lib = CDLL(platform_libname('metis'))
        except OSError:
            raise RuntimeError('Unable to load metis')

        # Constants
        self.METIS_NOPTIONS = 40

        # METIS_SetDefaultOptions
        self.METIS_SetDefaultOptions = lib.METIS_SetDefaultOptions
        self.METIS_SetDefaultOptions.argtypes = [c_void_p]
        self.METIS_SetDefaultOptions.errcheck = self._errcheck

        # METIS_PartGraphKway
        self.METIS_PartGraphKway = lib.METIS_PartGraphKway
        self.METIS_PartGraphKway.argtypes = [
            POINTER(c_int64), POINTER(c_int64),
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
            POINTER(c_int64), c_void_p, c_void_p, c_void_p,
            POINTER(c_int64), c_void_p
        ]
        self.METIS_PartGraphKway.errcheck = self._errcheck

        # METIS_PartGraphRecursive
        self.METIS_PartGraphRecursive = lib.METIS_PartGraphRecursive
        self.METIS_PartGraphRecursive.argtypes = [
            POINTER(c_int64), POINTER(c_int64),
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
            POINTER(c_int64), c_void_p, c_void_p, c_void_p,
            POINTER(c_int64), c_void_p
        ]
        self.METIS_PartGraphRecursive.errcheck = self._errcheck

    def _errcheck(self, status, fn, args):
        if status != 1:
            try:
                raise self._statuses[status]
            except KeyError:
                raise METISError


class METISPartitioner(BasePartitioner):
    name = 'metis'

    def __init__(self, *args, **kwargs):
        super(METISPartitioner, self).__init__(*args, **kwargs)

        # Load METIS
        self._wrappers = METISWrappers()

    def _partition_graph(self, vtab, etab, partwts):
        w = self._wrappers

        # Type conversion
        vtab = np.asanyarray(vtab, dtype=np.int64)
        etab = np.asanyarray(etab, dtype=np.int64)
        partwts = np.array(partwts, dtype=np.float32)

        # Normalise the weights
        partwts /= np.sum(partwts)

        # Allocate the partition array
        parts = np.empty(len(vtab) - 1, dtype=np.int64)

        # METIS option array
        opts = np.empty(w.METIS_NOPTIONS, dtype=np.int64)
        w.METIS_SetDefaultOptions(opts.ctypes)

        # Integer parameters
        nvert, nconst = c_int64(len(vtab) - 1), c_int64(1)
        npart, objval = c_int64(len(partwts)), c_int64()

        # Partition
        w.METIS_PartGraphRecursive(
            nvert, nconst, vtab.ctypes, etab.ctypes, None, None, None, npart,
            partwts.ctypes, None, opts.ctypes, objval, parts.ctypes
        )

        return parts
