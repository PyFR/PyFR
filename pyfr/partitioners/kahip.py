from ctypes import POINTER, c_bool, c_double, c_int, c_void_p

import numpy as np

from pyfr.ctypesutil import LibWrapper
from pyfr.partitioners.base import BasePartitioner


class KaHIPWrappers(LibWrapper):
    _libname = 'kahip'

    # Functions
    _functions = [
        (None, 'kaffpa_balance', POINTER(c_int), c_void_p, c_void_p, c_void_p,
         c_void_p, POINTER(c_int), POINTER(c_double), c_bool, c_bool, c_int,
         c_int, c_void_p, c_void_p)
    ]


class KaHIPPartitioner(BasePartitioner):
    name = 'kahip'
    has_part_weights = False
    has_multiple_constraints = False

    # Integer options
    int_opts = {'seed', 'ufactor'}

    # Enumeration options
    enum_opts = {
        'perfect': {'false': 0, 'true': 1},
        'mode': {'fast': 0, 'eco': 1, 'strong': 2,
                 'fastsocial': 3, 'ecosocial': 4, 'strongsocial': 5}
    }

    # Default options
    dflt_opts = {'seed': 2079, 'ufactor': 10, 'perfect': 'false',
                 'mode': 'strong'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load KaHIP
        self._wrappers = KaHIPWrappers()

    def _partition_graph(self, graph, partwts):
        # Type conversion
        vtab = np.asanyarray(graph.vtab, dtype=np.int32)
        etab = np.asanyarray(graph.etab, dtype=np.int32)
        vwts = np.asanyarray(graph.vwts, dtype=np.int32)
        ewts = np.asanyarray(graph.ewts, dtype=np.int32)

        # Output partition array and edge cut array (unused)
        parts = np.empty(len(vtab) - 1, dtype=np.int32)
        ecuts = np.empty_like(parts)

        # Permitted imbalance
        imbalance = c_double(self.opts['ufactor'] / 1000)

        # Integer parameters
        nvert = c_int(len(vtab) - 1)
        nparts = c_int(len(partwts))

        self._wrappers.kaffpa_balance(
            nvert, vwts.ctypes, vtab.ctypes, ewts.ctypes, etab.ctypes,
            nparts, imbalance, self.opts['perfect'], True,
            self.opts['seed'], self.opts['mode'], ecuts.ctypes, parts.ctypes
        )

        return parts
