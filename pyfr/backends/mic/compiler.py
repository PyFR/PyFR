# -*- coding: utf-8 -*-

import itertools as it
import os
import tempfile

from pytools.prefork import call_capture_output

from pyfr.util import rm


class MICSourceModule(object):
    _dir_seq = it.count()

    def __init__(self, src, dev, cfg):
        # Create a scratch directory
        tmpidx = next(self._dir_seq)
        tmpdir = tempfile.mkdtemp(prefix='pyfr-{0}-'.format(tmpidx))

        # Find MKL
        mklroot = cfg.get('backend-mic', 'mkl-root', '$MKLROOT')

        try:
            # File names
            cn, ln = 'tmp.c', 'libtmp.so'

            # Write the source code out
            with open(os.path.join(tmpdir, cn), 'w') as f:
                f.write(src)

            # Compile and link
            cmd = [
                'icc',
                '-shared',                        # Create a shared library
                '-std=c99',                       # Enable C99 support
                '-Ofast',                         # Optimise
                '-mmic',                          # Compile for the MIC
                '-fopenmp',                       # Enable OpenMP support
                '-L{0}/lib/mic'.format(mklroot),  # MKL stuff
                '-lmkl_intel_lp64',               #  ...
                '-lmkl_core',                     #  ...
                '-lmkl_intel_thread',             #  ...
                '-fPIC',                          # Position-independent code
                '-o', ln, cn
            ]
            call_capture_output(cmd, cwd=tmpdir)

            # Load
            self._mod = dev.load_library(os.path.join(tmpdir, ln))
        finally:
            rm(tmpdir)

    def function(self, name, argtypes, restype=None):
        return getattr(self._mod, name)
