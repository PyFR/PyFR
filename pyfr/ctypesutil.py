 # -*- coding: utf-8 -*-

import sys


def platform_libname(ln):
    # Mac OS X
    if sys.platform == 'darwin':
        return 'lib%s.dylib' % ln
    # Windows
    elif sys.platform == 'Windows':
        return '%s.dll' % ln
    # Assume UNIX/Linux
    else:
        return 'lib%s.so' % ln
