# -*- coding: utf-8 -*-

from mpi4py import MPI

import pycuda.driver as cuda

from pyfr.backends.base import ComputeKernel, MPIKernel
from pyfr.backends.cuda.provider import CUDAKernelProvider, get_grid_for_block
from pyfr.backends.cuda.types import CUDAMPIView
from pyfr.nputil import npdtype_to_ctype


class CUDAPackingKernels(CUDAKernelProvider):
    def _sendrecv(self, mv, mpipreqfn, pid, tag):
        # If we are an MPI view then extract the MPI matrix
        mpimat = mv.mpimat if isinstance(mv, CUDAMPIView) else mv

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

        # Get the CUDA pack/unpack kernel from the pack module
        fn = self._get_function('pack', 'pack_view', 'iiiPPPPP',
                                dict(dtype=npdtype_to_ctype(m.dtype)))

        # Compute the grid and thread-block size
        block = (128, 1, 1)
        grid = get_grid_for_block(block, v.n)

        # Create a CUDA event
        event = cuda.Event(cuda.event_flags.DISABLE_TIMING)

        class PackMPIViewKernel(ComputeKernel):
            def run(self, scomp, scopy):
                cstrides = getattr(v, 'cstrides', 0)
                rstrides = getattr(v, 'rstrides', 0)

                # Pack
                fn.prepared_async_call(grid, block, scomp, v.n, v.nvrow,
                                       v.nvcol, v.basedata, v.mapping,
                                       cstrides, rstrides, m)

                # Copy the packed buffer to the host
                event.record(scomp)
                scopy.wait_for_event(event)
                cuda.memcpy_dtoh_async(m.hdata, m.data, scopy)

        return PackMPIViewKernel()

    def send_pack(self, mv, pid, tag):
        return self._sendrecv(mv, MPI.COMM_WORLD.Send_init, pid, tag)

    def recv_pack(self, mv, pid, tag):
        return self._sendrecv(mv, MPI.COMM_WORLD.Recv_init, pid, tag)

    def unpack(self, mv):
        class UnpackMPIMatrixKernel(ComputeKernel):
            def run(self, scomp, scopy):
                cuda.memcpy_htod_async(mv.data, mv.hdata, scomp)

        return UnpackMPIMatrixKernel()
