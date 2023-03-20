from ctypes import Structure, addressof, byref, c_int, c_void_p, cast
from functools import cached_property

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, Kernel,
                                MetaKernel)
from pyfr.backends.openmp.generator import OpenMPKernelGenerator
from pyfr.nputil import npdtype_to_ctypestype
from pyfr.util import memoize


class OpenMPKernel(Kernel):
    def __init__(self, mats=[], views=[], misc=[], kernel=None):
        super().__init__(mats, views, misc)

        if kernel:
            self.kernel = kernel

    def add_to_graph(self, graph, dnodes):
        graph.klist.append(self.kernel)

        return len(graph.klist)

    def run(self):
        self.kernel()


class _OpenCLMetaKernelCommon(MetaKernel):
    def add_to_graph(self, graph, dnodes):
        for k in self.kernels:
            k.add_to_graph(graph, dnodes)

        return len(graph.klist)


class OpenMPOrderedMetaKernel(_OpenCLMetaKernelCommon): pass
class OpenMPUnorderedMetaKernel(_OpenCLMetaKernelCommon): pass


class OpenMPKRunArgs(Structure):
    _fields_ = [('fun', c_void_p), ('args', c_void_p), ('nblocks', c_int)]


class OpenMPKernelFunction:
    def __init__(self, backend, fun, argcls):
        self.krunner = backend.krunner
        self.fun = fun
        self.kargs = argcls()
        self.nblocks = -1

    @cached_property
    def runargs(self):
        return OpenMPKRunArgs(cast(self.fun, c_void_p), addressof(self.kargs),
                              self.nblocks)

    def set_arg(self, i, v):
        setattr(self.kargs, f'arg{i}', getattr(v, '_as_parameter_', v))

    def set_args(self, *args, start=0):
        for i, arg in enumerate(args, start=start):
            self.set_arg(i, arg)

    def set_nblocks(self, nblocks):
        self.nblocks = nblocks

    def __call__(self):
        self.krunner(0, 1, byref(self.runargs))


class OpenMPKernelProvider(BaseKernelProvider):
    @memoize
    def _get_arg_cls(self, argtypes):
        fields = [(f'arg{i}', npdtype_to_ctypestype(arg))
                  for i, arg in enumerate(argtypes)]

        return type('ArgStruct', (Structure,), {'_fields_': fields})

    @memoize
    def _build_library(self, src):
        return self.backend.compiler.build(src)

    def _build_function(self, name, src, argtypes, restype=None):
        lib = self._build_library(src)
        return lib.function(name, restype,
                            [npdtype_to_ctypestype(arg) for arg in argtypes])

    def _build_kernel(self, name, src, argtypes):
        lib = self._build_library(src)
        fun = lib.function(name)

        return OpenMPKernelFunction(self.backend, fun,
                                    self._get_arg_cls(tuple(argtypes)))


class OpenMPPointwiseKernelProvider(OpenMPKernelProvider,
                                    BasePointwiseKernelProvider):
    kernel_generator_cls = OpenMPKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst, argmv):
        rtargs = []

        # Set the number of blocks
        fun.set_nblocks(-(-dims[-1] // self.backend.csubsz))

        # Process the arguments
        for i, k in enumerate(arglst):
            if isinstance(k, str):
                rtargs.append((i, k))
            else:
                fun.set_arg(i, k)

        class PointwiseKernel(OpenMPKernel):
            if rtargs:
                def bind(self, **kwargs):
                    for i, k in rtargs:
                        self.kernel.set_arg(i, kwargs[k])

        return PointwiseKernel(*argmv, kernel=fun)
