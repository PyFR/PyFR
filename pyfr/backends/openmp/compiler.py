# -*- coding: utf-8 -*-

from ctypes import CDLL
import itertools as it
import os
import platform
import shlex
import tempfile
import uuid

from appdirs import user_cache_dir
from pytools.prefork import call_capture_output

from pyfr.ctypesutil import platform_libname
from pyfr.nputil import npdtype_to_ctypestype
from pyfr.util import digest, lazyprop, mv, rm


class SourceModule(object):
    _dir_seq = it.count()

    def __init__(self, src, cfg):
        # Find GCC (or a compatible alternative)
        self.cc = cfg.getpath('backend-openmp', 'cc', 'cc')

        # User specified compiler flags
        self.cflags = shlex.split(cfg.get('backend-openmp', 'cflags', ''))

        # Get the processor string
        proc = platform.processor()

        # Get the compiler version string
        version = call_capture_output([self.cc, '-v'])

        # Get the base compiler command strig
        cmd = self.cc_cmd(None, None)

        # Compute a digest of the current processor, compiler, and source
        self.digest = digest(proc, version, cmd, src)

        # Attempt to load the library from the cache
        self.mod = self._cache_loadlib()

        # Otherwise, we need to compile the kernel
        if not self.mod:
            # Create a scratch directory
            tmpidx = next(self._dir_seq)
            tmpdir = tempfile.mkdtemp(prefix='pyfr-{0}-'.format(tmpidx))

            try:
                # Compile and link the source into a shared library
                cname, lname = 'tmp.c', platform_libname('tmp')

                # Write the source code out
                with open(os.path.join(tmpdir, cname), 'w') as f:
                    f.write(src)

                # Invoke the compiler
                call_capture_output(self.cc_cmd(cname, lname), cwd=tmpdir)

                # Determine the fully qualified library name
                lpath = os.path.join(tmpdir, lname)

                # Add it to the cache and load
                self.mod = self._cache_set_and_loadlib(lpath)
            finally:
                # Unless we're debugging delete the scratch directory
                if 'PYFR_DEBUG_OMP_KEEP_LIBS' not in os.environ:
                    rm(tmpdir)

    def cc_cmd(self, srcname, libname):
        cmd = [
            self.cc,                # Compiler name
            '-shared',              # Create a shared library
            '-std=c99',             # Enable C99 support
            '-Ofast',               # Optimise, incl. -ffast-math
            '-march=native',        # Use CPU-specific instructions
            '-fopenmp',             # Enable OpenMP support
            '-fPIC',                # Generate position-independent code
            '-o', libname, srcname, # Library and source file names
            '-lm'                   # Link against libm
        ]

        # Append any user-provided arguments and return
        return cmd + self.cflags

    @lazyprop
    def cachedir(self):
        return os.environ.get('PYFR_OMP_CACHE_DIR',
                              user_cache_dir('pyfr', 'pyfr'))

    def _cache_loadlib(self):
        # If caching is disabled then return
        if 'PYFR_DEBUG_OMP_DISABLE_CACHE' in os.environ:
            return
        # Otherwise, check the cache
        else:
            # Determine the cached library name
            clname = platform_libname(self.digest)

            # Attempt to load the library
            try:
                return CDLL(os.path.join(self.cachedir, clname))
            except OSError:
                return

    def _cache_set_and_loadlib(self, lpath):
        # If caching is disabled then just load the library as-is
        if 'PYFR_DEBUG_OMP_DISABLE_CACHE' in os.environ:
            return CDLL(lpath)
        # Otherwise, move the library into the cache and load
        else:
            # Determine the cached library name and path
            clname = platform_libname(self.digest)
            clpath = os.path.join(self.cachedir, clname)
            ctpath = os.path.join(self.cachedir, str(uuid.uuid4()))

            try:
                # Ensure the cache directory exists
                os.makedirs(self.cachedir, exist_ok=True)

                # Perform a two-phase move to get the library in place
                mv(lpath, ctpath)
                mv(ctpath, clpath)
            # If an exception is raised, load from the original path
            except OSError:
                return CDLL(lpath)
            # Otherwise, load from the cache dir
            else:
                return CDLL(clpath)

    def function(self, name, restype, argtypes):
        # Get the function
        fn = getattr(self.mod, name)
        fn.restype = npdtype_to_ctypestype(restype)
        fn.argtypes = [npdtype_to_ctypestype(a) for a in argtypes]

        return fn
