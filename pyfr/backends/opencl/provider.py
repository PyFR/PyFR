from pyfr.backends.base import (BaseKernelProvider, BaseOrderedMetaKernel,
                                BasePointwiseKernelProvider,
                                BaseUnorderedMetaKernel, Kernel)
from pyfr.backends.opencl.generator import OpenCLKernelGenerator
from pyfr.cache import memoize
from pyfr.nputil import npdtype_to_ctypestype


class OpenCLKernel(Kernel):
    def add_to_graph(self, graph, deps):
        pass


class OpenCLOrderedMetaKernel(BaseOrderedMetaKernel):
    def add_to_graph(self, graph, deps):
        pass

    def run(self, queue, wait_for=None, ret_evt=False):
        for k in self.kernels[:-1]:
            wait_for = [k.run(queue, wait_for, True)]

        return self.kernels[-1].run(queue, wait_for, ret_evt)


class OpenCLUnorderedMetaKernel(BaseUnorderedMetaKernel):
    def add_to_graph(self, graph, deps):
        pass

    def run(self, queue, wait_for=None, ret_evt=False):
        if ret_evt:
            kevts = [k.run(queue, wait_for, True) for k in self.kernels]
            return queue.marker(kevts)
        else:
            for k in self.kernels:
                k.run(queue, wait_for, False)


class OpenCLKernelProvider(BaseKernelProvider):
    def _benchmark(self, kfunc, nbench=4, nwarmup=1):
        queue = self.backend.cl.queue(profiling=True)

        for i in range(nbench + nwarmup):
            if i == nwarmup:
                start_evt = end_evt = kfunc(queue)
            elif i == nbench + nwarmup - 1:
                end_evt = kfunc(queue)
            else:
                kfunc(queue)

        queue.finish()

        return (end_evt.end_time - start_evt.start_time) / nbench

    @memoize
    def _build_program(self, src):
        flags = ['-cl-fast-relaxed-math', '-cl-std=CL2.0']

        return self.backend.compiler.build(src, flags)

    def _build_kernel(self, name, src, argtypes, argn=[]):
        argtypes = [npdtype_to_ctypestype(arg) for arg in argtypes]

        return self._build_program(src).get_kernel(name, argtypes)


class OpenCLPointwiseKernelProvider(OpenCLKernelProvider,
                                    BasePointwiseKernelProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ls1d = (64,)
        self._ls2d = (64, 4)

        # Pass the local work group sizes to the generator
        class KernelGenerator(OpenCLKernelGenerator):
            block1d = self._ls1d
            block2d = self._ls2d

        self.kernel_generator_cls = KernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst, argm, argv):
        rtargs = []

        # Determine the work group sizes
        if len(dims) == 1:
            ls = self._ls1d
            gs = (dims[0] - dims[0] % -ls[0],)
        else:
            ls = self._ls2d
            gs = (dims[1] - dims[1] % -ls[0], ls[1])

        fun.set_dims(gs, ls)

        # Process the arguments
        for i, k in enumerate(arglst):
            if isinstance(k, str):
                rtargs.append((i, k))
            else:
                fun.set_arg(i, k)

        class PointwiseKernel(OpenCLKernel):
            if rtargs:
                def bind(self, **kwargs):
                    for i, k in rtargs:
                        fun.set_arg(i, kwargs[k])

            def run(self, queue, wait_for=None, ret_evt=False):
                return fun.exec_async(queue, wait_for, ret_evt)

        return PointwiseKernel(argm, argv)
