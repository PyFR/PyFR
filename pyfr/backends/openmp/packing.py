# -*- coding: utf-8 -*-

from mpi4py import MPI

from pyfr.backends.base import ComputeKernel, MPIKernel
from pyfr.backends.openmp.provider import OpenMPKernelProvider
from pyfr.backends.openmp.types import OpenMPMPIMatrix, OpenMPMPIView
from pyfr.nputil import npdtype_to_ctype


class OpenMPPackingKernels(OpenMPKernelProvider):
    def _sendrecv(self, mv, mpipreqfn, pid, tag):
        # If we are an MPI view then extract the MPI matrix
        mpimat = mv.mpimat if isinstance(mv, OpenMPMPIView) else mv

        # Create a persistent MPI request to send/recv the pack
        preq = mpipreqfn(mpimat.data, pid, tag)

        class SendRecvPackKernel(MPIKernel):
            def run(self, reqlist):
                # Start the request and append us to the list of requests
                preq.Start()
                reqlist.append(preq)

        return SendRecvPackKernel()

    def pack(self, mv):
        # An MPI view is simply a regular view plus an MPI matrix
        m, v = mv.mpimat, mv.view

        fn = self._get_function('pack', 'pack_view', None, 'iiiPPPPP',
                                dict(dtype=npdtype_to_ctype(m.dtype)))

        cstrides = getattr(v, 'cstrides', 0)
        rstrides = getattr(v, 'rstrides', 0)

        return self._basic_kernel(fn, v.n, v.nvrow, v.nvcol, v.basedata,
                                  v.mapping, cstrides, rstrides, m)

    def send_pack(self, mv, pid, tag):
        return self._sendrecv(mv, MPI.COMM_WORLD.Send_init, pid, tag)

    def recv_pack(self, mv, pid, tag):
        return self._sendrecv(mv, MPI.COMM_WORLD.Recv_init, pid, tag)

    def unpack(self, mv):
        # No-op
        class UnpackMPIMatrixKernel(ComputeKernel):
            def run(self):
                pass

        return UnpackMPIMatrixKernel()
