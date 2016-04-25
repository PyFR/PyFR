# -*- coding: utf-8 -*-

import ctypes
import ctypes.util
import os
import sys


def find_libc():
    if sys.platform == 'win32':
        return ctypes.util.find_msvcrt()
    else:
        return ctypes.util.find_library('c')


def load_library(name):
    lname = platform_libname(name)
    sdirs = platform_libdirs()

    # First attempt to utilise the system search path
    try:
        return ctypes.CDLL(lname)
    # Otherwise, if this fails then run our own search
    except OSError:
        for sd in sdirs:
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
