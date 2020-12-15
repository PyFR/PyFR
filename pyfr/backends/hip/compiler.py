# -*- coding: utf-8 -*-

import itertools as it
import os
import shlex
import tempfile
import uuid

from appdirs import user_cache_dir
from pytools.prefork import call_capture_output

from pyfr.nputil import npdtype_to_ctypestype
from pyfr.util import digest, lazyprop, mv, rm


class SourceModule(object):
    _dir_seq = it.count()

    def __init__(self, backend, src):
        self.backend = backend

        # Find HIPCC
        self.hipcc = backend.cfg.getpath('backend-hip', 'hipcc', 'hipcc')

        # User specified compiler flags
        self.cflags = shlex.split(backend.cfg.get('backend-hip', 'cflags', ''))

        # Get the compiler version string
        version = call_capture_output([self.hipcc, '--version'])

        # Get the base compiler command string
        cmd = self.cc_cmd(None, None)

        # Prepare the source code
        src = f'#include "hip/hip_runtime.h"\nextern "C"\n{{\n{src}\n}}'

        # Compute a digest of the compiler and source
        self.digest = digest(version, cmd, src)

        # Attempt to load the module from the cache
        self.mod = self._cache_loadmod()

        # Otherwise, we need to compile the kernel
        if not self.mod:
            # Create a scratch directory
            tmpidx = next(self._dir_seq)
            tmpdir = tempfile.mkdtemp(prefix=f'pyfr-{tmpidx}-')

            try:
                # Source and module file names
                cname, mname = 'tmp.cpp', 'tmp.mod'

                # Write the source code out
                with open(os.path.join(tmpdir, cname), 'w') as f:
                    f.write(src)

                # Invoke the compiler
                call_capture_output(self.cc_cmd(cname, mname), cwd=tmpdir)

                # Determine the fully qualified module path
                mpath = os.path.join(tmpdir, mname)

                # Add it to the cache and load
                self.mod = self._cache_set_and_loadmod(mpath)
            finally:
                # Unless we're debugging delete the scratch directory
                if 'PYFR_DEBUG_HIP_KEEP_LIBS' not in os.environ:
                    rm(tmpdir)

    def cc_cmd(self, srcname, modname):
        cmd = [
            self.hipcc,             # Compiler name
            '--genco',              # Generate a kernel
            '-Ofast',               # Optimise, incl. -ffast-math
            '-o', modname, srcname, # Module and source file names
        ]

        # Append any user-provided arguments and return
        return cmd + self.cflags

    @lazyprop
    def cachedir(self):
        return os.environ.get('PYFR_HIP_CACHE_DIR',
                              user_cache_dir('pyfr', 'pyfr'))

    def _cache_loadmod(self):
        # If caching is disabled then return
        if 'PYFR_DEBUG_HIP_DISABLE_CACHE' in os.environ:
            return
        # Otherwise, check the cache
        else:
            # Determine the cached module name and path
            cmname = f'{self.digest}.mod'
            cmpath = os.path.join(self.cachedir, cmname)

            # Attempt to load the module
            try:
                return self.backend.hip.load_module(cmpath)
            except Exception:
                return

    def _cache_set_and_loadmod(self, mpath):
        # If caching is disabled then just load the module as-is
        if 'PYFR_DEBUG_HIP_DISABLE_CACHE' in os.environ:
            return self.backend.hip.load_module(mpath)
        # Otherwise, move the module into the cache and load
        else:
            # Determine the cached library name and path
            cmname = f'{self.digest}.mod'
            cmpath = os.path.join(self.cachedir, cmname)
            ctpath = os.path.join(self.cachedir, str(uuid.uuid4()))

            try:
                # Ensure the cache directory exists
                os.makedirs(self.cachedir, exist_ok=True)

                # Perform a two-phase move to get the module in place
                mv(mpath, ctpath)
                mv(ctpath, cmpath)
            # If an exception is raised, load from the original path
            except OSError:
                return self.backend.hip.load_module(mpath)
            # Otherwise, load from the cache dir
            else:
                return self.backend.hip.load_module(cmpath)

    def function(self, name, argtypes):
        argtypes = [npdtype_to_ctypestype(arg) for arg in argtypes]

        return self.mod.get_function(name, argtypes)
