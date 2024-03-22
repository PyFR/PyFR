from ctypes import (POINTER, byref, c_double, c_int, c_int32, c_int64, c_float,
                    c_void_p)

import numpy as np

from pyfr.ctypesutil import LibWrapper
from pyfr.partitioners.base import BasePartitioner
from pyfr.util import silence


# Possible METIS exception types
class METISError(Exception): pass
class METISErrorInput(METISError): pass
class METISErrorMemory(METISError): pass


class METISWrappers(LibWrapper):
    _libname = 'metis'

    # Error codes
    _statuses = {
        -2: METISErrorInput,
        -3: METISErrorMemory,
        '*': METISError
    }

    # Success status code
    _status_noerr = 1

    # Constants
    NOPTIONS = 40
    OPTION_PTYPE = 0
    OPTION_CTYPE = 2
    OPTION_IPTYPE = 3
    OPTION_RTYPE = 4
    OPTION_NIPARTS = 6
    OPTION_NITER = 7
    OPTION_NCUTS = 8
    OPTION_SEED = 9
    OPTION_MINCONN = 11
    OPTION_NSEPS = 16
    OPTION_UFACTOR = 17

    def _load_library(self):
        lib = super()._load_library()

        # Attempt to determine the integer type used by METIS
        opts = np.arange(0, self.NOPTIONS, dtype=np.int64)
        lib.METIS_SetDefaultOptions(opts.ctypes)

        # If the last element was set then assume 64-bit ints
        if opts[-1] != self.NOPTIONS - 1:
            self.metis_int = metis_int = c_int64
            self.metis_int_np = metis_int_np = np.int64
        # Otherwise go with 32-bits
        else:
            self.metis_int = metis_int = c_int32
            self.metis_int_np = metis_int_np = np.int32

        def intref(v=0): return byref(metis_int(v))

        # Attempt to partition a two vertex graph into two partitions
        vtab = np.array([0, 1, 2], dtype=metis_int_np)
        etab = np.array([1, 0], dtype=metis_int_np)
        pwts = np.array([0.5, 0.5], dtype=np.float64)
        prts = np.empty(len(vtab) - 1, dtype=metis_int_np)

        with silence():
            ret = lib.METIS_PartGraphKway(
                intref(len(vtab) - 1), intref(1), vtab.ctypes, etab.ctypes,
                None, None, None, intref(len(pwts)), pwts.ctypes, None, None,
                intref(), prts.ctypes
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

        return lib

    @property
    def _functions(self):
        return [
            (c_int, 'METIS_SetDefaultOptions', c_void_p),
            (c_int, 'METIS_PartGraphKway', POINTER(self.metis_int),
             POINTER(self.metis_int), c_void_p, c_void_p, c_void_p, c_void_p,
             c_void_p, POINTER(self.metis_int), c_void_p, c_void_p, c_void_p,
             POINTER(self.metis_int), c_void_p),
            (c_int, 'METIS_PartGraphRecursive', POINTER(self.metis_int),
             POINTER(self.metis_int), c_void_p, c_void_p, c_void_p, c_void_p,
             c_void_p, POINTER(self.metis_int), c_void_p, c_void_p, c_void_p,
             POINTER(self.metis_int), c_void_p)
        ]


class METISPartitioner(BasePartitioner):
    name = 'metis'
    has_part_weights = True
    has_multiple_constraints = True

    # Integer options
    int_opts = {'niter', 'niparts', 'ncuts', 'seed', 'nseps', 'ufactor'}

    # Enumeration options
    enum_opts = {
        'ptype': {'rb': 0, 'kway': 1},
        'ctype': {'rm': 0, 'shem': 1},
        'iptype': {'grow': 0, 'random': 1, 'edge': 2, 'node': 3, 'metisrb': 4},
        'rtype': {'fm': 0, 'greedy': 1, 'sep2sided': 3, 'sep1sided': 4},
        'minconn': {'false': 0, 'true': 1}
    }

    # Default options
    dflt_opts = {'seed': 2079, 'ufactor': 10, 'ptype': 'rb', 'minconn': 'true'}

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

        # Prepare the partition weights
        partwts = np.array(partwts, dtype=w.metis_flt_np)
        partwts /= np.sum(partwts)

        # Account for multiple partitioning constraints
        partwts = np.repeat(partwts[:, None], vwts.shape[1], axis=1)

        # Allocate the partition array
        parts = np.empty(len(vtab) - 1, dtype=w.metis_int_np)

        # Allocate our options array
        opts = np.empty(w.NOPTIONS, dtype=w.metis_int_np)
        w.METIS_SetDefaultOptions(opts.ctypes)

        # Process our options
        for k, v in self.opts.items():
            opts[getattr(w, f'OPTION_{k.upper()}')] = v

        # Select the partitioning function
        if self.opts['ptype'] == self.enum_opts['ptype']['rb']:
            part_graph_fn = w.METIS_PartGraphRecursive
        else:
            part_graph_fn = w.METIS_PartGraphKway

        # Integer parameters
        nvert, nconst = w.metis_int(len(vtab) - 1), w.metis_int(vwts.shape[1])
        npart, objval = w.metis_int(len(partwts)), w.metis_int()

        # Partition
        with silence():
            part_graph_fn(
                nvert, nconst, vtab.ctypes, etab.ctypes, vwts.ctypes, None,
                ewts.ctypes, npart, partwts.ctypes, None, opts.ctypes, objval,
                parts.ctypes
            )

        # Check for invalid partition numbers
        if np.max(parts) >= len(partwts):
            raise RuntimeError('Invalid partition number from METIS')

        return parts.astype(np.int32)
