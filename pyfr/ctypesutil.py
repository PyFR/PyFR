# -*- coding: utf-8 -*-

import ctypes
import ctypes.util
import os
import sys


def get_libc_function(fn):
    if sys.platform == 'win32':
        if sys.version_info.minor >= 5:
            libc = ctypes.windll.msvcrt
        else:
            libc = ctypes.CDLL(ctypes.util.find_msvcrt())
    else:
        libc = ctypes.CDLL(ctypes.util.find_library('c'))

    return getattr(libc, fn)


def load_library(name):
    # If an explicit override has been given then use it
    lpath = os.environ.get('PYFR_{0}_LIBRARY_PATH'.format(name.upper()))
    if lpath:
        return ctypes.CDLL(lpath)

    # Otherwise synthesise the library name and start searching
    lname = platform_libname(name)

    # Start with system search path
    try:
        return ctypes.CDLL(lname)
    # ..and if this fails then run our own search
    except OSError:
        for sd in platform_libdirs():
            try:
                return ctypes.CDLL(os.path.abspath(os.path.join(sd, lname)))
            except OSError:
                pass
        else:
            raise OSError('Unable to load {0}'.format(name))


def platform_libname(name):
    if sys.platform == 'darwin':
        return 'lib{0}.dylib'.format(name)
    elif sys.platform == 'win32':
        return '{0}.dll'.format(name)
    else:
        return 'lib{0}.so'.format(name)


def platform_libdirs():
    path = os.environ.get('PYFR_LIBRARY_PATH', '')
    dirs = [d for d in path.split(':') if d]

    # On Mac OS X append the default path used by MacPorts
    if sys.platform == 'darwin':
        return dirs + ['/opt/local/lib']
    # Otherwise just return
    else:
        return dirs
