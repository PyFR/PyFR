# -*- coding: utf-8 -*-

from ctypes import POINTER, c_int, c_int32, c_int64, c_double, c_void_p

import numpy as np

from pyfr.ctypesutil import LibWrapper
from pyfr.partitioners.base import BasePartitioner


# Possible Scotch exception types
class SCOTCHError(Exception): pass


class SCOTCHWrappers(LibWrapper):
    _libname = 'scotch'

    # Error codes
    _statuses = {
        '*': SCOTCHError
    }

    # Types
    SCOTCH_Arch = c_double*128
    SCOTCH_Graph = c_double*128
    SCOTCH_Strat = c_double*128

    # Functions
    _functions = [
        (c_int, 'SCOTCH_archInit', POINTER(SCOTCH_Arch)),
        (c_int, 'SCOTCH_graphInit', POINTER(SCOTCH_Graph)),
        (c_int, 'SCOTCH_stratInit', POINTER(SCOTCH_Strat)),
        (c_int, 'SCOTCH_graphMap', POINTER(SCOTCH_Graph), POINTER(SCOTCH_Arch),
         POINTER(SCOTCH_Strat), c_void_p),
        (None, 'SCOTCH_archExit', POINTER(SCOTCH_Arch)),
        (None, 'SCOTCH_graphExit', POINTER(SCOTCH_Graph)),
        (None, 'SCOTCH_stratExit', POINTER(SCOTCH_Strat))
    ]

    def __init__(self):
        super().__init__()

        # Ascertain the integer size
        if self._lib.SCOTCH_numSizeof() == 4:
            self.scotch_int = scotch_int = c_int32
            self.scotch_int_np = np.int32
        else:
            self.scotch_int = scotch_int = c_int64
            self.scotch_int_np = np.int64

        # SCOTCH_archCmpltw
        self.SCOTCH_archCmpltw = self._lib.SCOTCH_archCmpltw
        self.SCOTCH_archCmpltw.argtypes = [
            POINTER(self.SCOTCH_Arch), scotch_int, c_void_p
        ]
        self.SCOTCH_archCmpltw.errcheck = self._errcheck

        # SCOTCH_graphBuild
        self.SCOTCH_graphBuild = self._lib.SCOTCH_graphBuild
        self.SCOTCH_graphBuild.argtypes = [
            POINTER(self.SCOTCH_Graph), scotch_int, scotch_int, c_void_p,
            c_void_p, c_void_p, c_void_p, scotch_int, c_void_p, c_void_p
        ]
        self.SCOTCH_graphBuild.errcheck = self._errcheck

        # SCOTCH_stratGraphMapBuild
        self.SCOTCH_stratGraphMapBuild = self._lib.SCOTCH_stratGraphMapBuild
        self.SCOTCH_stratGraphMapBuild.argtypes = [
            POINTER(self.SCOTCH_Strat), scotch_int, scotch_int, c_double
        ]
        self.SCOTCH_stratGraphMapBuild.errcheck = self._errcheck


class SCOTCHPartitioner(BasePartitioner):
    name = 'scotch'

    # Interger options
    int_opts = {'ufactor'}

    # Enumeration options
    enum_opts = {
        'strat': {'default': 0, 'quality': 1, 'speed': 2, 'balance': 4}
    }

    # Default options
    dflt_opts = {'ufactor': 10, 'strat': 'default'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load SCOTCH
        self._wrappers = SCOTCHWrappers()

    def _partition_graph(self, graph, partwts):
        w = self._wrappers

        # Type conversion
        vtab = np.asanyarray(graph.vtab, dtype=w.scotch_int)
        etab = np.asanyarray(graph.etab, dtype=w.scotch_int)
        vwts = np.asanyarray(graph.vwts, dtype=w.scotch_int)
        ewts = np.asanyarray(graph.ewts, dtype=w.scotch_int)
        partwts = np.asanyarray(partwts, dtype=w.scotch_int)

        # Output partition array
        parts = np.empty(len(vtab) - 1, dtype=w.scotch_int)

        # Allocate
        arch = w.SCOTCH_Arch()
        graph = w.SCOTCH_Graph()
        strat = w.SCOTCH_Strat()

        try:
            # Initialise
            w.SCOTCH_archInit(arch)
            w.SCOTCH_graphInit(graph)
            w.SCOTCH_stratInit(strat)

            # Apply the partition weights
            w.SCOTCH_archCmpltw(arch, len(partwts), partwts.ctypes)

            # Construct the graph
            w.SCOTCH_graphBuild(
                graph, 0, len(vtab) - 1, vtab.ctypes, None, vwts.ctypes, None,
                len(etab), etab.ctypes, ewts.ctypes
            )

            # Permitted load imbalance ratio
            balrat = self.opts['ufactor'] / 1000.0

            # Partitioning stratergy
            w.SCOTCH_stratGraphMapBuild(
                strat, self.opts['strat'], len(partwts), balrat
            )

            # Perform the partitioning
            w.SCOTCH_graphMap(graph, arch, strat, parts.ctypes)
        finally:
            if any(v != 0.0 for v in arch):
                w.SCOTCH_archExit(arch)

            if any(v != 0.0 for v in graph):
                w.SCOTCH_graphExit(graph)

            if any(v != 0.0 for v in strat):
                w.SCOTCH_stratExit(strat)

        return parts
