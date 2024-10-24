from ctypes import CDLL
import itertools as it
import os
from pathlib import Path
import platform
import shlex
import tempfile

from pytools.prefork import call_capture_output

from pyfr.cache import ObjectCache
from pyfr.ctypesutil import platform_libname
from pyfr.util import digest, rm


class OpenMPCompiler:
    _dir_seq = it.count()

    def __init__(self, cfg):
        # Find GCC (or a compatible alternative)
        self.cc = cfg.getpath('backend-openmp', 'cc', 'cc')

        # User specified compiler flags
        self.cflags = shlex.split(cfg.get('backend-openmp', 'cflags', ''))

        # Get the processor string
        self.proc = platform.processor()

        # Get the compiler version string
        self.version = call_capture_output([self.cc, '-v'])

        # Get the base compiler command strig
        self.cmd = self.cc_cmd(None, None)

        # Get the cache
        self.cache = ObjectCache('omp')

    def build(self, src):
        # Compute a digest of the current processor, compiler, and source
        ckey = digest(self.proc, self.version, self.cmd, src)

        # Attempt to load the library from the cache
        mod = self._cache_loadlib(ckey)

        # Otherwise, we need to compile the kernel
        if mod is None:
            # Create a scratch directory
            tmpidx = next(self._dir_seq)
            tmpdir = Path(tempfile.mkdtemp(prefix=f'pyfr-{tmpidx}-'))

            try:
                # Temporary source and library names
                cname, lname = 'tmp.c', platform_libname('tmp')

                # Write the source code out
                (tmpdir / cname).write_bytes(src.encode())

                # Invoke the compiler
                call_capture_output(self.cc_cmd(cname, lname), cwd=tmpdir)

                # Add it to the cache and load it
                mod = self._cache_set_and_loadlib(ckey, tmpdir / lname)
            finally:
                # Unless we're debugging delete the scratch directory
                if 'PYFR_DEBUG_OMP_KEEP_LIBS' not in os.environ:
                    rm(tmpdir)

        return OpenMPCompilerModule(mod)

    def cc_cmd(self, srcname, libname):
        cmd = [
            self.cc,                # Compiler name
            '-shared',              # Create a shared library
            '-std=c11',             # Enable C11 support
            '-Ofast',               # Optimise, incl. -ffast-math
            '-march=native',        # Use CPU-specific instructions
            '-fopenmp',             # Enable OpenMP support
            '-fPIC',                # Generate position-independent code
            '-o', libname, srcname, # Library and source file names
            '-lm'                   # Link against libm
        ]

        # Append any user-provided arguments and return
        return cmd + self.cflags

    def _cache_loadlib(self, ckey):
        if path := self.cache.get_path(platform_libname(ckey)):
            try:
                return CDLL(path)
            except OSError:
                return

    def _cache_set_and_loadlib(self, ckey, lpath):
        ckey = platform_libname(ckey)

        # Attempt to add the item to the cache and load
        if cpath := self.cache.set_with_path(ckey, lpath):
            return CDLL(cpath)
        # Otherwise, load from the current temporary path
        else:
            return CDLL(lpath)


class OpenMPCompilerModule:
    def __init__(self, mod):
        self.mod = mod

    def function(self, name, restype=None, argtypes=None):
        fn = getattr(self.mod, name)
        fn.restype = restype
        if argtypes:
            fn.argtypes = argtypes

        return fn
