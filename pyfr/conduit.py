from ctypes import (RTLD_GLOBAL, c_char_p, c_double, c_int, c_int32, c_int64,
                    c_void_p)

import numpy as np

from pyfr.ctypesutil import LibWrapper


class ConduitError(Exception): pass


_conduit_functions = [
    (c_int, 'conduit_datatype_sizeof_index_t'),
    (c_void_p, 'conduit_node_append', c_void_p),
    (c_void_p, 'conduit_node_create', c_void_p),
    (None, 'conduit_node_destroy', c_void_p),
    (None, 'conduit_node_set_path_char8_str', c_void_p, c_char_p, c_char_p),
    (None, 'conduit_node_set_path_float32_ptr', c_void_p, c_char_p,
     c_void_p, c_int64),
    (None, 'conduit_node_set_path_float64', c_void_p, c_char_p, c_double),
    (None, 'conduit_node_set_path_float64_ptr', c_void_p, c_char_p,
     c_void_p, c_int64),
    (None, 'conduit_node_set_path_int32', c_void_p, c_char_p, c_int32),
    (None, 'conduit_node_set_path_int64', c_void_p, c_char_p, c_int64),
    (None, 'conduit_node_set_path_int64_ptr', c_void_p, c_char_p,
     c_void_p, c_int64),
    (None, 'conduit_node_set_path_node', c_void_p, c_char_p, c_void_p),
    # Strided external variants for interleaved (AoS) multi-component arrays
    (None, 'conduit_node_set_path_external_float32_ptr_detailed',
     c_void_p, c_char_p, c_void_p, c_int64, c_int64, c_int64, c_int64, c_int64),
    (None, 'conduit_node_set_path_external_float64_ptr_detailed',
     c_void_p, c_char_p, c_void_p, c_int64, c_int64, c_int64, c_int64, c_int64),
]


class ConduitWrappers(LibWrapper):
    _libname = 'conduit'
    _errtype = c_void_p
    _mode = RTLD_GLOBAL
    _functions = _conduit_functions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.conduit_datatype_sizeof_index_t() != 8:
            raise RuntimeError('Conduit must be compiled with 64-bit index '
                               'types')

    def _errcheck(self, status, fn, args):
        if not status:
            raise ConduitError

        return status


class ConduitNode:
    def __init__(self, lib, ptr=None, child=False):
        self.lib = lib
        self.child = child
        self._as_parameter_ = ptr or self.lib.conduit_node_create(None)

    def __del__(self):
        if not self.child:
            self.lib.conduit_node_destroy(self)

    def __setitem__(self, key, value):
        key = key.encode()
        match value:
            case str():
                self.lib.conduit_node_set_path_char8_str(self, key,
                                                         value.encode())
            case ConduitNode():
                self.lib.conduit_node_set_path_node(self, key, value)
            case int():
                if value.bit_length() <= 32:
                    self.lib.conduit_node_set_path_int32(self, key, value)
                else:
                    self.lib.conduit_node_set_path_int64(self, key, value)
            case float():
                self.lib.conduit_node_set_path_float64(self, key, value)
            case (np.ndarray() | np.generic()):
                value = np.ascontiguousarray(value)
                fn = getattr(self.lib,
                             f'conduit_node_set_path_{value.dtype}_ptr')
                fn(self, key, value.ctypes.data, value.size)
            case list():
                value = np.array(value, dtype=float)
                self.lib.conduit_node_set_path_float64_ptr(self, key,
                                                           value.ctypes.data,
                                                           value.size)
            case _:
                raise ValueError('ConduitNode: __setitem__ type not supported')

    def set_aos(self, key, labels, arr2d):
        # Set columns of a C-contiguous (npoints, ncomps) array as interleaved
        # external references so VTK creates an AoS array (no GetVoidPointer
        # copy).  The caller must keep arr2d alive until the next
        # catalyst_execute call.
        npoints, ncomps = arr2d.shape
        itemsize = arr2d.itemsize
        stride = ncomps * itemsize
        fn = getattr(self.lib,
                     f'conduit_node_set_path_external_{arr2d.dtype}_ptr_detailed')
        for i, l in enumerate(labels):
            fn(self, f'{key}/{l}'.encode(), arr2d.ctypes.data,
               npoints, i * itemsize, stride, itemsize, 0)

    def append(self):
        ptr = self.lib.conduit_node_append(self)
        return ConduitNode(self.lib, ptr, child=True)
