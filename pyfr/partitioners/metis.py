# -*- coding: utf-8 -*-

from ctypes import CDLL, POINTER, c_int64, c_float, c_void_p

import numpy as np

from pyfr.ctypesutil import load_library
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
        lib = load_library('metis')

        # Relevant constants
        self.METIS_NOPTIONS = 40
        self.METIS_OPTION_PTYPE = 0
        self.METIS_OPTION_CTYPE = 2
        self.METIS_OPTION_IPTYPE = 3
        self.METIS_OPTION_RTYPE = 4
        self.METIS_OPTION_NITER = 6
        self.METIS_OPTION_NCUTS = 7
        self.METIS_OPTION_NSEPS = 15
        self.METIS_OPTION_UFACTOR = 16
        self.METIS_OPTION_MINCONN = 9

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

    # Integer options
    int_opts = {'niter', 'ncuts', 'nseps', 'ufactor'}

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
        super(METISPartitioner, self).__init__(*args, **kwargs)

        # Load METIS
        self._wrappers = METISWrappers()

    def _partition_graph(self, graph, partwts):
        w = self._wrappers

        # Type conversion
        vtab = np.asanyarray(graph.vtab, dtype=np.int64)
        etab = np.asanyarray(graph.etab, dtype=np.int64)
        vwts = np.asanyarray(graph.vwts, dtype=np.int64)
        ewts = np.asanyarray(graph.ewts, dtype=np.int64)
        partwts = np.array(partwts, dtype=np.float32)

        # Normalise the weights
        partwts /= np.sum(partwts)

        # Allocate the partition array
        parts = np.empty(len(vtab) - 1, dtype=np.int64)

        # Allocate our options array
        opts = np.empty(w.METIS_NOPTIONS, dtype=np.int64)
        w.METIS_SetDefaultOptions(opts.ctypes)

        # Process our options
        for k, v in self.opts.iteritems():
            oidx = getattr(w, 'METIS_OPTION_' + k.upper())
            opts[oidx] = v

        # Select the partitioning function
        if self.opts['ptype'] == 'rb':
            part_graph_fn = w.METIS_PartGraphRecursive
        else:
            part_graph_fn = w.METIS_PartGraphKway

        # Integer parameters
        nvert, nconst = c_int64(len(vtab) - 1), c_int64(1)
        npart, objval = c_int64(len(partwts)), c_int64()

        # Partition
        part_graph_fn(
            nvert, nconst, vtab.ctypes, etab.ctypes, vwts.ctypes, None,
            ewts.ctypes, npart, partwts.ctypes, None, opts.ctypes, objval,
            parts.ctypes
        )

        return parts
