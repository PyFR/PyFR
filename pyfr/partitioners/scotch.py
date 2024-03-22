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
    SCOTCH_Context = c_double*128
    SCOTCH_Graph = c_double*128
    SCOTCH_Strat = c_double*128

    # Constants
    OPTION_DETERMINISTIC = 0

    def _load_library(self):
        lib = super()._load_library()

        if lib.SCOTCH_numSizeof() == 4:
            self.scotch_int = c_int32
            self.scotch_int_np = np.int32
        else:
            self.scotch_int = c_int64
            self.scotch_int_np = np.int64

        return lib

    @property
    def _functions(self):
        return [
            (c_int, 'SCOTCH_archInit', POINTER(self.SCOTCH_Arch)),
            (c_int, 'SCOTCH_archCmpltw', POINTER(self.SCOTCH_Arch),
             self.scotch_int, c_void_p),
            (None, 'SCOTCH_archExit', POINTER(self.SCOTCH_Arch)),
            (c_int, 'SCOTCH_contextInit', POINTER(self.SCOTCH_Context)),
            (c_int, 'SCOTCH_contextBindGraph', POINTER(self.SCOTCH_Context),
             POINTER(self.SCOTCH_Graph), POINTER(self.SCOTCH_Graph)),
            (c_int, 'SCOTCH_contextOptionSetNum', POINTER(self.SCOTCH_Context),
             c_int, self.scotch_int),
            (None, 'SCOTCH_contextRandomSeed', POINTER(self.SCOTCH_Context),
             self.scotch_int),
            (None, 'SCOTCH_contextExit', POINTER(self.SCOTCH_Context)),
            (c_int, 'SCOTCH_graphInit', POINTER(self.SCOTCH_Graph)),
            (c_int, 'SCOTCH_graphBuild', POINTER(self.SCOTCH_Graph),
             self.scotch_int, self.scotch_int, c_void_p, c_void_p, c_void_p,
             c_void_p, self.scotch_int, c_void_p, c_void_p),
            (c_int, 'SCOTCH_graphMap', POINTER(self.SCOTCH_Graph),
             POINTER(self.SCOTCH_Arch), POINTER(self.SCOTCH_Strat), c_void_p),
            (None, 'SCOTCH_graphExit', POINTER(self.SCOTCH_Graph)),
            (c_int, 'SCOTCH_stratInit', POINTER(self.SCOTCH_Strat)),
            (c_int, 'SCOTCH_stratGraphMapBuild', POINTER(self.SCOTCH_Strat),
             self.scotch_int, self.scotch_int, c_double),
            (None, 'SCOTCH_stratExit', POINTER(self.SCOTCH_Strat))
        ]


class SCOTCHPartitioner(BasePartitioner):
    name = 'scotch'
    has_part_weights = True
    has_multiple_constraints = False

    # Interger options
    int_opts = {'ufactor', 'seed'}

    # Enumeration options
    enum_opts = {
        'strat': {'default': 0, 'quality': 1, 'speed': 2, 'balance': 4}
    }

    # Default options
    dflt_opts = {'seed': 2079, 'ufactor': 10, 'strat': 'default'}

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
        context = w.SCOTCH_Context()
        graph_org = w.SCOTCH_Graph()
        graph_ctx = w.SCOTCH_Graph()
        strat = w.SCOTCH_Strat()

        try:
            # Apply the partition weights
            w.SCOTCH_archInit(arch)
            w.SCOTCH_archCmpltw(arch, len(partwts), partwts.ctypes)

            # Construct the origin graph
            w.SCOTCH_graphInit(graph_org)
            w.SCOTCH_graphBuild(
                graph_org, 0, len(vtab) - 1, vtab.ctypes, None, vwts.ctypes,
                None, len(etab), etab.ctypes, ewts.ctypes
            )

            # Permitted load imbalance ratio
            balrat = self.opts['ufactor'] / 1000

            # Partitioning stratergy
            w.SCOTCH_stratInit(strat)
            w.SCOTCH_stratGraphMapBuild(
                strat, self.opts['strat'], len(partwts), balrat
            )

            # Configure the partitioning context
            w.SCOTCH_contextInit(context)
            w.SCOTCH_contextOptionSetNum(context, w.OPTION_DETERMINISTIC, 1)
            w.SCOTCH_contextRandomSeed(context, self.opts['seed'])

            # Bind this context to our origin graph to obtain
            w.SCOTCH_graphInit(graph_ctx)
            w.SCOTCH_contextBindGraph(context, graph_org, graph_ctx)

            # Perform the partitioning
            w.SCOTCH_graphMap(graph_ctx, arch, strat, parts.ctypes)
        finally:
            if any(v != 0.0 for v in strat):
                w.SCOTCH_stratExit(strat)

            if any(v != 0.0 for v in graph_ctx):
                w.SCOTCH_graphExit(graph_ctx)

            if any(v != 0.0 for v in context):
                w.SCOTCH_contextExit(context)

            if any(v != 0.0 for v in graph_org):
                w.SCOTCH_graphExit(graph_org)

            if any(v != 0.0 for v in arch):
                w.SCOTCH_archExit(arch)

        return parts.astype(np.int32)
