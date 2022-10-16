import ctypes
import ctypes.util
import os
import sys


class LibWrapper:
    _libname = None
    _statuses = None
    _functions = None
    _mode = ctypes.DEFAULT_MODE

    def __init__(self):
        self._lib = lib = load_library(self._libname, self._mode)

        for fret, fname, *fargs in self._functions:
            fn = getattr(lib, fname)
            fn.restype = fret
            fn.argtypes = fargs

            if fret == ctypes.c_int:
                fn.errcheck = self._errcheck

            setattr(self, self._transname(fname), fn)

    def _transname(self, fname):
        return fname

    def _errcheck(self, status, fn, args):
        if status != 0:
            try:
                raise self._statuses[status]
            except KeyError:
                raise self._statuses['*'] from None


def get_libc_function(fn):
    if sys.platform == 'win32':
        libc = ctypes.windll.msvcrt
    else:
        libc = ctypes.CDLL(ctypes.util.find_library('c'))

    return getattr(libc, fn)


def load_library(name, mode=ctypes.DEFAULT_MODE):
    # If an explicit override has been given then use it
    lpath = os.environ.get(f'PYFR_{name.upper()}_LIBRARY_PATH')
    if lpath:
        return ctypes.CDLL(lpath, mode=mode)

    # Otherwise synthesise the library name and start searching
    lname = platform_libname(name)

    # Check our search paths
    for sd in platform_libdirs():
        try:
            return ctypes.CDLL(os.path.abspath(os.path.join(sd, lname)),
                               mode=mode)
        except OSError:
            pass

    # …and if this fails then defer to the system search path
    return ctypes.CDLL(lname, mode=mode)


def platform_libname(name):
    if sys.platform == 'darwin':
        return f'lib{name}.dylib'
    elif sys.platform == 'win32':
        return f'{name}.dll'
    else:
        return f'lib{name}.so'


def platform_libdirs():
    path = os.environ.get('PYFR_LIBRARY_PATH', '')
    dirs = [d for d in path.split(':') if d]

    # On macOS append the default path used by MacPorts and Homebrew
    if sys.platform == 'darwin':
        return dirs + ['/opt/local/lib', '/opt/homebrew/lib']
    # Otherwise just return
    else:
        return dirs
