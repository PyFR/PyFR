from gimmik import OpenCLMatMul
import numpy as np

from pyfr.backends.base import NotSuitableError
from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider


class OpenCLGiMMiKKernels(OpenCLKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        # Maximum number of non-zeros
        self.max_nnz = backend.cfg.getint('backend-opencl', 'gimmik-max-nnz',
                                          2048)

        # Maximum number of kernels to consider
        self.nkerns = backend.cfg.getint('backend-opencl', 'gimmik-nkerns', 8)

        # Number of benchmarking runs
        self.nbench = backend.cfg.getint('backend-opencl', 'gimmik-nbench', 5)

        # Kernel cache
        self._kerns = {}

        # Queue used for benchmarking
        self._queue = backend.cl.queue(profiling=True)

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('GiMMiK requires a constant a matrix')

        # Fetch the matrix
        arr = a.get()

        # Check that A is reasonably sparse
        if np.count_nonzero(arr) > self.max_nnz:
            raise NotSuitableError('Matrix too dense for GiMMiK')

        # Dimensions
        ldb, ldc = b.leaddim, out.leaddim

        # Alignment
        if 'align' in b.tags and 'align' in out.tags:
            aligne = self.backend.alignb // b.itemsize
        else:
            aligne = None

        # Cache key
        ckey = (a.mid, alpha, beta, aligne, ldb, ldc)

        # Check the kernel cache
        try:
            kern, gs, ls = self._kerns[ckey]

            # Clone the kernel so it gets its own set of arguments
            kern = kern.clone()
        except KeyError:
            kname = f'gimmik_mm_{arr.shape[0]}x{arr.shape[1]}'
            queue = self._queue
            local_mem_size = self.backend.cl.dev.local_mem_size
            best_dt, best_kern = None, None
            sdata = None

            # Save a copy of the contents of the output matrix
            out_np = getattr(out, 'parent', out).get()

            mm = OpenCLMatMul(alpha*arr, beta=beta, aligne=aligne, n=b.ncol,
                              ldb=ldb, ldc=ldc)
            kgen = mm.kernels(a.dtype, kname=kname,
                              local_mem_size=local_mem_size)

            # Benchmark the sequence of kernels generated by GiMMiK
            try:
                for i in range(self.nkerns):
                    src, meta = kgen.send(sdata)

                    gs, ls = meta['global_work_size'], meta['local_work_size']
                    kern = self._build_kernel(kname, src, 'PP')

                    # Set the parameters
                    kern.set_dims(gs, ls)
                    kern.set_args(b, out)

                    # Benchmark with warmup
                    for j in range(self.nbench + 1):
                        if j == 1:
                            start_evt = end_evt = kern.exec_async(queue,
                                                                  ret_evt=True)
                        elif j == self.nbench:
                            end_evt = kern.exec_async(queue, ret_evt=True)
                        else:
                            kern.exec_async(queue)

                    queue.finish()
                    dt = end_evt.end_time - start_evt.start_time
                    if best_dt is None or dt < best_dt:
                        best_dt = dt
                        best_kern = kern, gs, ls

                    sdata = {'runtime': dt}
            except StopIteration:
                pass

            # Restore the output matrix
            getattr(out, 'parent', out).set(out_np)

            # Update the cache
            self._kerns[ckey] = kern, gs, ls = best_kern

        # Set the parameters
        kern.set_dims(gs, ls)
        kern.set_args(b, out)

        class MulKernel(OpenCLKernel):
            def run(self, queue, wait_for=None, ret_evt=False):
                return kern.exec_async(queue, wait_for, ret_evt)

        return MulKernel(mats=[b, out])
