# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

from pyfr.backends.base import ComputeKernel, MPIKernel
from pyfr.backends.opencl.provider import OpenCLKernelProvider
from pyfr.backends.opencl.types import OpenCLMPIView


class OpenCLPackingKernels(OpenCLKernelProvider):
    def _sendrecv(self, mv, mpipreqfn, pid, tag):
        # If we are an MPI view then extract the MPI matrix
        mpimat = mv.mpimat if isinstance(mv, OpenCLMPIView) else mv

        # Create a persistent MPI request to send/recv the pack
        preq = mpipreqfn(mpimat.hdata, pid, tag)

        class SendRecvPackKernel(MPIKernel):
            def run(self, reqlist):
                # Start the request and append us to the list of requests
                preq.Start()
                reqlist.append(preq)

        return SendRecvPackKernel()

    def pack(self, mv):
        # An MPI view is simply a regular view plus an MPI matrix
        m, v = mv.mpimat, mv.view

        # Render the kernel template
        tpl = self.backend.lookup.get_template('pack')
        src = tpl.render(alignb=self.backend.alignb, fpdtype=m.dtype)

        # Build
        kern = self._build_kernel('pack_view', src, [np.int32]*3 + [np.intp]*5)

        class PackMPIViewKernel(ComputeKernel):
            def run(self, qcomp, qcopy):
                # Kernel arguments
                args = [v.n, v.nvrow, v.nvcol, v.basedata, v.mapping,
                        v.cstrides, v.rstrides, m]
                args = [getattr(arg, 'data', arg) for arg in args]

                # Pack
                event = kern(qcomp, (v.n,), None, *args)

                # Copy the packed buffer to the host
                cl.enqueue_copy(qcopy, m.hdata, m.data, is_blocking=False,
                                wait_for=[event])

        return PackMPIViewKernel()

    def send_pack(self, mv, pid, tag):
        from mpi4py import MPI

        return self._sendrecv(mv, MPI.COMM_WORLD.Send_init, pid, tag)

    def recv_pack(self, mv, pid, tag):
        from mpi4py import MPI

        return self._sendrecv(mv, MPI.COMM_WORLD.Recv_init, pid, tag)

    def unpack(self, mv):
        class UnpackMPIMatrixKernel(ComputeKernel):
            def run(self, qcomp, qcopy):
                cl.enqueue_copy(qcomp, mv.data, mv.hdata, is_blocking=False)

        return UnpackMPIMatrixKernel()
