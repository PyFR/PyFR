from ctypes import c_char_p, c_int

from pyfr.ctypesutil import LibWrapper


class ROCTXWrappers(LibWrapper):
    _libname = 'rocprofiler-sdk-roctx'

    _functions = [
        (c_int, 'roctxRangePushA', c_char_p),
        (c_int, 'roctxRangePop'),
    ]


class ROCTXAnnotator:
    def __init__(self):
        self.lib = ROCTXWrappers()

    def push(self, name):
        self.lib.roctxRangePushA(name.encode())

    def pop(self, name):
        self.lib.roctxRangePop()
