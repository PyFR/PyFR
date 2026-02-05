import numpy as np

from pyfr.backends.base import BaseBackend
from pyfr.mpiutil import get_local_rank


class OpenCLBackend(BaseBackend):
    name = 'opencl'
    blocks = False

    def __init__(self, cfg):
        super().__init__(cfg)

        from pyfr.backends.opencl.compiler import OpenCLCompiler
        from pyfr.backends.opencl.driver import OpenCL

        # Load and wrap OpenCL
        self.cl = OpenCL()

        # Get the platform/device info from the config file
        platid = cfg.get('backend-opencl', 'platform-id', '0').lower()
        devid = cfg.get('backend-opencl', 'device-id', 'local-rank').lower()
        devtype = cfg.get('backend-opencl', 'device-type', 'all').upper()

        # Handle the local-rank case
        if devid == 'local-rank':
            devid = str(get_local_rank())

        # Determine the OpenCL platform to use
        for i, platform in enumerate(self.cl.get_platforms()):
            if platid == str(i) or platid == platform.name.lower():
                break
        else:
            raise ValueError('No suitable OpenCL platform found')

        # Determine the OpenCL device to use
        for i, device in enumerate(platform.get_devices(devtype)):
            if (devid == str(i) or devid == device.name.lower() or
                devid == str(device.uuid or '')):
                break
        else:
            raise ValueError('No suitable OpenCL device found')

        # Determine if the device supports double precision arithmetic
        if self.fpdtype == np.float64 and not device.has_fp64:
            raise ValueError('Device does not support double precision')

        # Set the device
        self.cl.set_device(device)

        # OpenCL compiler
        self.compiler = OpenCLCompiler(self.cl)

        # Compute the alignment requirement for the context
        self.alignb = device.mem_align

        # Compute the SoA size
        self.soasz = 2*self.alignb // np.dtype(self.fpdtype).itemsize
        self.csubsz = self.soasz

        from pyfr.backends.opencl import (blasext, clblast, gimmik, packing,
                                          provider, tinytc, types)

        # Register our data types and meta kernels
        self.const_matrix_cls = types.OpenCLConstMatrix
        self.graph_cls = types.OpenCLGraph
        self.matrix_cls = types.OpenCLMatrix
        self.matrix_slice_cls = types.OpenCLMatrixSlice
        self.view_cls = types.OpenCLView
        self.xchg_matrix_cls = types.OpenCLXchgMatrix
        self.xchg_view_cls = types.OpenCLXchgView
        self.ordered_meta_kernel_cls = provider.OpenCLOrderedMetaKernel
        self.unordered_meta_kernel_cls = provider.OpenCLUnorderedMetaKernel

        # Instantiate the base kernel providers
        kprovs = [provider.OpenCLPointwiseKernelProvider,
                  blasext.OpenCLBlasExtKernels,
                  packing.OpenCLPackingKernels,
                  gimmik.OpenCLGiMMiKKernels]
        self._providers = [k(self) for k in kprovs]

        # Load CLBlast if available
        try:
            self._providers.append(clblast.OpenCLCLBlastKernels(self))
        except OSError:
            pass

        # Load TinyTC if available
        try:
            self._providers.append(tinytc.OpenCLTinyTCKernels(self))
        except OSError:
            pass

        # Pointwise kernels
        self.pointwise = self._providers[0]

        # Queues (in and out of order)
        self.queue = self.cl.queue(out_of_order=True)

    def run_kernels(self, kernels, wait=False):
        # Submit the kernels to the command queue
        for k in kernels:
            self.queue.barrier()
            k.run(self.queue)

        if wait:
            self.queue.finish()
        else:
            self.queue.flush()

    def run_graph(self, graph, wait=False):
        self.queue.barrier()

        graph.run(self.queue)

        if wait:
            self.queue.finish()

    def wait(self):
        self.queue.finish()

    def _malloc_impl(self, nbytes):
        # Allocate the device buffer
        buf = self.cl.mem_alloc(nbytes)

        # Zero the buffer
        self.cl.zero(buf, 0, nbytes)

        return buf
