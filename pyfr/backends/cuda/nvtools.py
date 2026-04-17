from ctypes import (POINTER, Structure, c_char_p, c_int, c_int32, c_uint16,
                    c_uint32, c_uint64, c_void_p, sizeof)

from pyfr.ctypesutil import LibWrapper


class NVTXEventAttributes(Structure):
    _fields_ = [
        ('version', c_uint16),
        ('size', c_uint16),
        ('category', c_uint32),
        ('colour_type', c_int32),
        ('colour', c_uint32),
        ('payload_type', c_int32),
        ('_reserved_0', c_int32),
        ('payload', c_uint64),
        ('message_type', c_int32),
        ('message', c_char_p),
    ]


class NVTXWrappers(LibWrapper):
    _libname = 'nvtx3interop'

    _functions = [
        (c_void_p, 'nvtxDomainCreateA', c_char_p),
        (None, 'nvtxDomainDestroy', c_void_p),
        (c_int, 'nvtxDomainRangePushEx', c_void_p,
         POINTER(NVTXEventAttributes)),
        (c_int, 'nvtxDomainRangePop', c_void_p),
    ]


class NVTXAnnotator:
    _palette = [
        0xFF00CC00, 0xFF0077B6, 0xFFEE6C4D, 0xFF7209B7,
        0xFFF4A261, 0xFF2EC4B6, 0xFFE63946, 0xFF457B9D,
    ]

    def __init__(self, name='PyFR'):
        self.lib = NVTXWrappers()
        self._as_parameter_ = self.lib.nvtxDomainCreateA(name.encode())
        self._regions = {}

    def __del__(self):
        if getattr(self, '_as_parameter_', None):
            self.lib.nvtxDomainDestroy(self)

    def _make_attr(self, name):
        self._regions[name] = attr = NVTXEventAttributes()
        attr.version = 2
        attr.size = sizeof(attr)
        attr.colour_type = 1
        attr.colour = self._palette[len(self._regions) % len(self._palette)]
        attr.message_type = 1
        attr.message = name.encode()

        return attr

    def push(self, name):
        attr = self._regions.get(name) or self._make_attr(name)
        self.lib.nvtxDomainRangePushEx(self, attr)

    def pop(self, name):
        self.lib.nvtxDomainRangePop(self)
