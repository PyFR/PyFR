from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from weakref import finalize

from gimmik import HIPMatMul

from pyfr.backends.base import NotSuitableError
from pyfr.backends.hip.compiler import HIPRTC
from pyfr.backends.hip.provider import HIPKernel, HIPKernelProvider
from pyfr.cache import ObjectCache
from pyfr.util import digest


_worker_cache = None
_worker_hiprtc = None


def _kernel_cache_entry(src, gcn_arch, compiler_version):
    src = f'extern "C"\n{{\n{src}\n}}'
    flags = [f'--gpu-architecture={gcn_arch}', '-munsafe-fp-atomics']
    ckey = digest(compiler_version, 'kernel', src, flags)

    return ckey, src, flags


def _warm_kernel_cache(src, gcn_arch, compiler_version):
    global _worker_cache, _worker_hiprtc

    if _worker_cache is None:
        _worker_cache = ObjectCache('hip')
    if _worker_hiprtc is None:
        _worker_hiprtc = HIPRTC()

    ckey, src, flags = _kernel_cache_entry(src, gcn_arch, compiler_version)

    if _worker_cache.get_bytes(ckey) is None:
        code = _worker_hiprtc.compile('kernel', src, flags)
        _worker_cache.set_with_bytes(ckey, code)


def _warmup_worker(_):
    global _worker_cache, _worker_hiprtc

    if _worker_cache is None:
        _worker_cache = ObjectCache('hip')
    if _worker_hiprtc is None:
        _worker_hiprtc = HIPRTC()

    return True


def _render_compile_candidate(args):
    (idx, arr, dtype, alpha, beta, aligne, kname, gcn_arch, warp_size,
     compiler_version) = args

    mm = HIPMatMul(alpha*arr, beta=beta, aligne=aligne)
    src, meta = mm.render_candidate(idx, dtype, kname=kname,
                                    gcn_arch=gcn_arch,
                                    warp_size=warp_size)

    _warm_kernel_cache(src, gcn_arch, compiler_version)

    return src, meta


class HIPGiMMiKKernels(HIPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        # Maximum number of kernels to consider
        self.nkerns = backend.cfg.getint('backend-hip', 'gimmik-nkerns', 20)

        # Number of benchmarking runs
        self.nbench = backend.cfg.getint('backend-hip', 'gimmik-nbench', 5)

        # Parallel source rendering and compiler-cache warmup
        self.parallel_compile = backend.cfg.getbool(
            'backend-hip', 'gimmik-parallel-compile', False
        )
        self.parallel_compile_workers = backend.cfg.getint(
            'backend-hip', 'gimmik-parallel-compile-workers', self.nkerns
        )
        self.parallel_compile_pool = None

        if self.parallel_compile:
            workers = max(1, self.parallel_compile_workers)
            mpctx = mp.get_context('spawn')
            self.parallel_compile_pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=mpctx
            )

            # Spawn workers eagerly so the first autotune does not pay for
            # process creation and HIPRTC initialization.
            list(self.parallel_compile_pool.map(_warmup_worker,
                                                range(workers)))

        # Kernel cache
        self._mul_kerns = {}

    def __del__(self):
        try:
            if self.parallel_compile_pool is not None:
                self.parallel_compile_pool.shutdown()
        except AttributeError:
            pass

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('GiMMiK requires a constant a matrix')

        # Fetch the matrix
        arr = a.get()

        # Dimensions
        n = b.ncol
        ldb, ldc = b.leaddim, out.leaddim

        # Alignment
        if 'align' in b.tags and 'align' in out.tags:
            aligne = self.backend.alignb // b.itemsize
        else:
            aligne = None

        # Cache key
        ckey = (a.mid, alpha, beta, aligne)

        # Check the kernel cache
        try:
            kern, block, grid_y, ncolsv, dt = self._mul_kerns[ckey]
        except KeyError:
            ifac = self.backend.autotune_ifac
            kname = f'gimmik_mm_{arr.shape[0]}x{arr.shape[1]}'
            kdata = None
            best_kern = None

            # Save a copy of the contents of the output matrix
            out_np = getattr(out, 'parent', out).get()

            def benchmark_candidate(src, meta):
                kern = self._build_kernel(kname, src, 'iPiPi')

                grid_y = meta.get('grid_y', 1)
                ncolsv = (
                    meta.get('width', 1) *
                    meta.get('ncols', meta['block'][0])
                )
                grid = (-(-n // ncolsv), grid_y, 1)
                params = kern.make_params(grid, meta['block'])
                params.set_args(n, b, ldb, out, ldc)

                # Obtain the runtime
                dt = self._benchmark(
                    lambda stream: kern.exec_async(stream, params),
                    nbench=self.nbench
                )

                kdata = {
                    'runtime': dt,
                    'registers': kern.nreg,
                    'local_mem': kern.local_mem
                }

                return kern, meta['block'], grid_y, ncolsv, dt, kdata

            def update_best(bench):
                nonlocal best_kern

                if best_kern is None or bench[4] < ifac*best_kern[4]:
                    best_kern = bench[:-1]

            mm = HIPMatMul(alpha*arr, beta=beta, aligne=aligne)

            if self.parallel_compile:
                gcn_arch = self.backend.props['gcn_arch_name']
                warp_size = self.backend.props['warp_size']
                count = min(
                    self.nkerns,
                    mm.candidate_count(a.dtype, gcn_arch=gcn_arch,
                                       warp_size=warp_size)
                )
                args = [
                    (
                        idx, arr, a.dtype, alpha, beta, aligne, kname,
                        gcn_arch, warp_size, self.backend.compiler.version
                    )
                    for idx in range(count)
                ]

                candidates = list(
                    self.parallel_compile_pool.map(
                        _render_compile_candidate, args
                    )
                ) if args else []

                for src, meta in candidates:
                    update_best(benchmark_candidate(src, meta))
            else:
                kgen = mm.kernels(
                    a.dtype, kname=kname,
                    gcn_arch=self.backend.props['gcn_arch_name'],
                    warp_size=self.backend.props['warp_size']
                )

                # Benchmark the sequence of kernels generated by GiMMiK
                try:
                    for i in range(self.nkerns):
                        src, meta = kgen.send(kdata)

                        bench = benchmark_candidate(src, meta)
                        update_best(bench)
                        kdata = bench[-1]
                except StopIteration:
                    pass

            # Restore the output matrix
            getattr(out, 'parent', out).set(out_np)

            # Update the cache
            self._mul_kerns[ckey] = (
                kern, block, grid_y, ncolsv, dt
            ) = best_kern
            finalize(a, lambda: self._mul_kerns.pop(ckey))

        # Set the parameters
        grid = (-(-n // ncolsv), grid_y, 1)
        params = kern.make_params(grid, block)
        params.set_args(n, b, ldb, out, ldc)

        class MulKernel(HIPKernel):
            def add_to_graph(self, graph, deps):
                return graph.graph.add_kernel(params, deps)

            def run(self, stream):
                kern.exec_async(stream, params)

        return MulKernel(mats=[a, b, out], dt=dt)
