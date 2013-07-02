# -*- coding: utf-8 -*-

from pyfr.backends.base import ComputeKernel
from pyfr.backends.openmp.compiler import GccSourceModule
from pyfr.template import PkgTemplateLookup
from pyfr.util import memoize


class OpenMPKernelProvider(object):
    lookup = PkgTemplateLookup(__name__, 'kernels')

    def __init__(self, backend, cfg):
        self._cfg = cfg

    @memoize
    def _get_module(self, module, tplparams={}):
        # Get the template file
        tpl = self.lookup.get_template(module + '.c.mak')

        # Filter floating point constants
        if tplparams['dtype'] == 'float':
            fpfilt = lambda v: v + ('f' if '.' in v or 'e' in v else '.f')
        else:
            fpfilt = lambda v: v

        # Render the template
        mod = tpl.render(f=fpfilt, **tplparams)

        # Compile
        return GccSourceModule(mod, self._cfg)

    @memoize
    def _get_function(self, module, function, restype, argtypes,
                      tplparams={}):
        # Compile off the module
        mod = self._get_module(module, tplparams)

        # Bind
        return mod.function(function, restype, argtypes)

    def _basic_kernel(self, fn, *args):
        class BasicKernel(ComputeKernel):
            def run(self):
                fn(*args)

        return BasicKernel()
