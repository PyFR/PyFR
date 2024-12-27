from ctypes import (POINTER, Structure, Union, addressof, byref, c_int,
                    c_void_p, cast, pointer, sizeof)
from functools import cached_property

from pyfr.backends.base import (BaseKernelProvider, BaseOrderedMetaKernel,
                                BasePointwiseKernelProvider,
                                BaseUnorderedMetaKernel, Kernel)
from pyfr.backends.openmp.generator import OpenMPKernelGenerator
from pyfr.cache import memoize
from pyfr.nputil import npdtype_to_ctypestype


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


class _OpenMPMetaKernel:
    def add_to_graph(self, graph, dnodes):
        for k in self.kernels:
            k.add_to_graph(graph, dnodes)

        return len(graph.klist)


class OpenMPOrderedMetaKernel(_OpenMPMetaKernel, BaseOrderedMetaKernel):
    pass


class OpenMPUnorderedMetaKernel(_OpenMPMetaKernel, BaseUnorderedMetaKernel):
    pass


class OpenMPRegularRunArgs(Structure):
    _fields_ = [('fun', c_void_p), ('args', c_void_p)]


class OpenMPBlockKernelArgs(Structure):
    _fields_ = [('fun', c_void_p),
                ('args', c_void_p),
                ('argmask', c_int),
                ('argsz', c_int),
                ('offset', c_int)]


class OpenMPBlockRunArgs(Structure):
    _fields_ = [('nkerns', c_int),
                ('nblocks', c_int),
                ('allocsz', c_int),
                ('nsubs', c_int),
                ('subs', POINTER(c_int)),
                ('kernels', POINTER(OpenMPBlockKernelArgs))]


class _OpenMPKRunArgsUnion(Union):
    _fields_ = [('r', OpenMPRegularRunArgs), ('b', OpenMPBlockRunArgs)]


class OpenMPKRunArgs(Structure):
    KTYPE_REGULAR = 0
    KTYPE_BLOCK_GROUP = 1

    _anonymous_ = ['u']
    _fields_ = [('ktype', c_int), ('u', _OpenMPKRunArgsUnion)]


class OpenMPKernelFunction:
    def __init__(self, backend, fun, argcls, argnames):
        self.krunner = backend.krunner
        self.fun = fun
        self.kargs = argcls()

        # Blocking info
        self.argnames = list(argnames)
        self.argsizes = [None]*len(argnames)
        self.nblocks = None
        self.subs_offsets = [0]*len(argnames)

    @cached_property
    def runargs(self):
        fun = cast(self.fun, c_void_p)
        args, argsz = addressof(self.kargs), sizeof(self.kargs)

        if self.nblocks is None:
            rra = OpenMPRegularRunArgs(fun=fun, args=args)
            return OpenMPKRunArgs(ktype=OpenMPKRunArgs.KTYPE_REGULAR, r=rra)
        else:
            bka = OpenMPBlockKernelArgs(fun=fun, args=args, argsz=argsz)
            bra = OpenMPBlockRunArgs(nkerns=1, nblocks=self.nblocks,
                                     kernels=pointer(bka))
            return OpenMPKRunArgs(ktype=OpenMPKRunArgs.KTYPE_BLOCK_GROUP,
                                  b=bra)

    def arg_off(self, i):
        return getattr(self.kargs.__class__, f'arg{i}').offset

    def arg_idx(self, name):
        return self.argnames.index(name)

    def arg_blocksz(self, i):
        return self.argsizes[i]

    def subs_off(self, i):
        return self.subs_offsets[i]

    def set_arg(self, i, v):
        setattr(self.kargs, f'arg{i}', getattr(v, '_as_parameter_', v))

        try:
            self.argsizes[i] = v.blocksz*v.itemsize
        except (AttributeError, IndexError):
            pass

        try:
            self.subs_offsets[i] = v.ra*v.leaddim*v.itemsize
        except (AttributeError, IndexError):
            pass

    def set_args(self, *args, start=0):
        for i, arg in enumerate(args, start=start):
            self.set_arg(i, arg)

    def set_nblocks(self, nblocks):
        self.nblocks = nblocks

    def __call__(self):
        self.krunner(1, byref(self.runargs))


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

    def _build_kernel(self, name, src, argtypes, argnames=[]):
        lib = self._build_library(src)
        fun = lib.function(name)

        return OpenMPKernelFunction(self.backend, fun,
                                    self._get_arg_cls(tuple(argtypes)),
                                    argnames)


class OpenMPPointwiseKernelProvider(OpenMPKernelProvider,
                                    BasePointwiseKernelProvider):
    kernel_generator_cls = OpenMPKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst, argm, argv):
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

        return PointwiseKernel(argm, argv, kernel=fun)
