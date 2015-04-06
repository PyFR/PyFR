# -*- coding: utf-8 -*-

from ctypes import POINTER, c_int, c_double, c_void_p

import numpy as np

from pyfr.ctypesutil import load_library
from pyfr.partitioners.base import BasePartitioner


class SCOTCHWrappers(object):
    def __init__(self):
        lib = load_library('scotch')

        # Types
        self.SCOTCH_Arch = SCOTCH_Arch = c_double*128
        self.SCOTCH_Graph = SCOTCH_Graph = c_double*128
        self.SCOTCH_Strat = SCOTCH_Strat = c_double*128

        # SCOTCH_archInit
        self.SCOTCH_archInit = lib.SCOTCH_archInit
        self.SCOTCH_archInit.argtypes = [POINTER(SCOTCH_Arch)]
        self.SCOTCH_archInit.errcheck = self._errcheck

        # SCOTCH_archExit
        self.SCOTCH_archExit = lib.SCOTCH_archExit
        self.SCOTCH_archExit.argtypes = [POINTER(SCOTCH_Arch)]
        self.SCOTCH_archExit.restype = None

        # SCOTCH_archCmpltw
        self.SCOTCH_archCmpltw = lib.SCOTCH_archCmpltw
        self.SCOTCH_archCmpltw.argtypes = [
            POINTER(SCOTCH_Arch), c_int, c_void_p
        ]
        self.SCOTCH_archCmpltw.errcheck = self._errcheck

        # SCOTCH_graphInit
        self.SCOTCH_graphInit = lib.SCOTCH_graphInit
        self.SCOTCH_graphInit.argtypes = [POINTER(SCOTCH_Graph)]
        self.SCOTCH_graphInit.errcheck = self._errcheck

        # SCOTCH_graphExit
        self.SCOTCH_graphExit = lib.SCOTCH_graphExit
        self.SCOTCH_graphExit.argtypes = [POINTER(SCOTCH_Graph)]
        self.SCOTCH_graphExit.restype = None

        # SCOTCH_graphBuild
        self.SCOTCH_graphBuild = lib.SCOTCH_graphBuild
        self.SCOTCH_graphBuild.argtypes = [
            POINTER(SCOTCH_Graph), c_int, c_int,
            c_void_p, c_void_p, c_void_p, c_void_p,
            c_int, c_void_p, c_void_p
        ]
        self.SCOTCH_graphBuild.errcheck = self._errcheck

        # SCOTCH_graphPart
        self.SCOTCH_graphMap = lib.SCOTCH_graphMap
        self.SCOTCH_graphMap.argtypes = [
            POINTER(SCOTCH_Graph), POINTER(SCOTCH_Arch),
            POINTER(SCOTCH_Strat), c_void_p
        ]
        self.SCOTCH_graphMap.errcheck = self._errcheck

        # SCOTCH_stratInit
        self.SCOTCH_stratInit = lib.SCOTCH_stratInit
        self.SCOTCH_stratInit.argtypes = [POINTER(SCOTCH_Strat)]
        self.SCOTCH_stratInit.errcheck = self._errcheck

        # SCOTCH_stratExit
        self.SCOTCH_stratExit = lib.SCOTCH_stratExit
        self.SCOTCH_stratExit.argtypes = [POINTER(SCOTCH_Strat)]
        self.SCOTCH_stratExit.restype = None

        # SCOTCH_stratGraphMapBuild
        self.SCOTCH_stratGraphMapBuild = lib.SCOTCH_stratGraphMapBuild
        self.SCOTCH_stratGraphMapBuild.argtypes = [
            POINTER(SCOTCH_Strat), c_int, c_int, c_double
        ]
        self.SCOTCH_stratGraphMapBuild.errcheck = self._errcheck

    def _errcheck(self, status, fn, args):
        if status != 0:
            raise RuntimeError('Scotch error')


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
        vtab = np.asanyarray(graph.vtab, dtype=np.int32)
        etab = np.asanyarray(graph.etab, dtype=np.int32)
        vwts = np.asanyarray(graph.vwts, dtype=np.int32)
        ewts = np.asanyarray(graph.ewts, dtype=np.int32)
        partwts = np.asanyarray(partwts, dtype=np.int32)

        # Output partition array
        parts = np.empty(len(vtab) - 1, dtype=np.int32)

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
