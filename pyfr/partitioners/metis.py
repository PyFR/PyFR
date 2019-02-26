# -*- coding: utf-8 -*-

from ctypes import (POINTER, byref, c_double, c_int32, c_int64, c_float,
                    c_void_p)

import numpy as np

from pyfr.ctypesutil import load_library
from pyfr.partitioners.base import BasePartitioner
from pyfr.util import silence


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

    # Constants
    METIS_NOPTIONS = 40
    METIS_OPTION_PTYPE = 0
    METIS_OPTION_CTYPE = 2
    METIS_OPTION_IPTYPE = 3
    METIS_OPTION_RTYPE = 4
    METIS_OPTION_NITER = 6
    METIS_OPTION_NCUTS = 7
    METIS_OPTION_SEED = 8
    METIS_OPTION_MINCONN = 10
    METIS_OPTION_NSEPS = 15
    METIS_OPTION_UFACTOR = 16

    def __init__(self):
        lib = load_library('metis')

        # Try to determine the data types METIS was compiled with
        with silence():
            self._probe_types(lib)

        # METIS_SetDefaultOptions
        self.METIS_SetDefaultOptions = lib.METIS_SetDefaultOptions
        self.METIS_SetDefaultOptions.argtypes = [c_void_p]
        self.METIS_SetDefaultOptions.errcheck = self._errcheck

        # METIS_PartGraphKway
        self.METIS_PartGraphKway = lib.METIS_PartGraphKway
        self.METIS_PartGraphKway.argtypes = [
            POINTER(self.metis_int), POINTER(self.metis_int),
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
            POINTER(self.metis_int), c_void_p, c_void_p, c_void_p,
            POINTER(self.metis_int), c_void_p
        ]
        self.METIS_PartGraphKway.errcheck = self._errcheck

        # METIS_PartGraphRecursive
        self.METIS_PartGraphRecursive = lib.METIS_PartGraphRecursive
        self.METIS_PartGraphRecursive.argtypes = [
            POINTER(self.metis_int), POINTER(self.metis_int),
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
            POINTER(self.metis_int), c_void_p, c_void_p, c_void_p,
            POINTER(self.metis_int), c_void_p
        ]
        self.METIS_PartGraphRecursive.errcheck = self._errcheck

    def _probe_types(self, lib):
        # Attempt to determine the integer type used by METIS
        opts = np.arange(0, self.METIS_NOPTIONS, dtype=np.int64)
        lib.METIS_SetDefaultOptions(opts.ctypes)

        # If the last element was set then assume 64-bit ints
        if opts[-1] != self.METIS_NOPTIONS - 1:
            self.metis_int = metis_int = c_int64
            self.metis_int_np = metis_int_np = np.int64
        # Otherwise go with 32-bits
        else:
            self.metis_int = metis_int = c_int32
            self.metis_int_np = metis_int_np = np.int32

        intref = lambda v=0: byref(metis_int(v))

        # Attempt to partition a two vertex graph into two partitions
        vtab = np.array([0, 1, 2], dtype=metis_int_np)
        etab = np.array([1, 0], dtype=metis_int_np)
        pwts = np.array([0.5, 0.5], dtype=np.float64)
        prts = np.empty(len(vtab) - 1, dtype=metis_int_np)

        ret = lib.METIS_PartGraphKway(
            intref(len(vtab) - 1), intref(1), vtab.ctypes, etab.ctypes, None,
            None, None, intref(len(pwts)), pwts.ctypes, None, None, intref(),
            prts.ctypes
        )

        # If successful then assume METIS is using 64-bit doubles
        if ret == 1:
            self.metis_flt, self.metis_flt_np = c_double, np.float64
        # Else if we are getting a parameter error assume 32-bit floats
        elif ret == -2:
            self.metis_flt, self.metis_flt_np = c_float, np.float32
        # Other, unknown error
        else:
            raise METISError

    def _errcheck(self, status, fn, args):
        if status != 1:
            try:
                raise self._statuses[status]
            except KeyError:
                raise METISError


class METISPartitioner(BasePartitioner):
    name = 'metis'

    # Integer options
    int_opts = {'niter', 'ncuts', 'seed', 'nseps', 'ufactor'}

    # Enumeration options
    enum_opts = {
        'ptype': {'rb': 0, 'kway': 1},
        'ctype': {'rm': 0, 'shem': 1},
        'iptype': {'grow': 0, 'random': 1, 'edge': 2, 'node': 3},
        'rtype': {'fm': 0, 'greedy': 1, 'sep2sided': 3, 'sep1sided': 4},
        'minconn': {'false': 0, 'true': 1}
    }

    # Default options
    dflt_opts = {'ufactor': 10, 'ptype': 'rb', 'minconn': 'true'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load METIS
        self._wrappers = METISWrappers()

    def _partition_graph(self, graph, partwts):
        w = self._wrappers

        # Type conversion
        vtab = np.asanyarray(graph.vtab, dtype=w.metis_int_np)
        etab = np.asanyarray(graph.etab, dtype=w.metis_int_np)
        vwts = np.asanyarray(graph.vwts, dtype=w.metis_int_np)
        ewts = np.asanyarray(graph.ewts, dtype=w.metis_int_np)
        partwts = np.array(partwts, dtype=w.metis_flt_np)

        # Normalise the weights
        partwts /= np.sum(partwts)

        # Allocate the partition array
        parts = np.empty(len(vtab) - 1, dtype=w.metis_int_np)

        # Allocate our options array
        opts = np.empty(w.METIS_NOPTIONS, dtype=w.metis_int_np)
        w.METIS_SetDefaultOptions(opts.ctypes)

        # Process our options
        for k, v in self.opts.items():
            oidx = getattr(w, 'METIS_OPTION_' + k.upper())
            opts[oidx] = v

        # Select the partitioning function
        if self.opts['ptype'] == self.enum_opts['ptype']['rb']:
            part_graph_fn = w.METIS_PartGraphRecursive
        else:
            part_graph_fn = w.METIS_PartGraphKway

        # Integer parameters
        nvert, nconst = w.metis_int(len(vtab) - 1), w.metis_int(1)
        npart, objval = w.metis_int(len(partwts)), w.metis_int()

        # Partition
        part_graph_fn(
            nvert, nconst, vtab.ctypes, etab.ctypes, vwts.ctypes, None,
            ewts.ctypes, npart, partwts.ctypes, None, opts.ctypes, objval,
            parts.ctypes
        )

        return parts
