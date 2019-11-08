# -*- coding: utf-8 -*-

from pyfr.backends.base.kernels import BaseKernelProvider, MPIKernel


class BasePackingKernels(BaseKernelProvider):
    def _sendrecv(self, mv, mpipreqfn, pid, tag):
        # If we are an exchange view then extract the exchange matrix
        if isinstance(mv, self.backend.xchg_view_cls):
            xchgmat = mv.xchgmat
        else:
            xchgmat = mv

        # Create a persistent MPI request to send/recv the matrix
        preq = mpipreqfn(xchgmat.hdata, pid, tag)

        class SendRecvPackKernel(MPIKernel):
            def run(self, queue):
                # Start the request and append us to the list of requests
                preq.Start()
                queue.mpi_reqs.append(preq)

        return SendRecvPackKernel()

    def pack(self, mv):
        pass

    def send_pack(self, mv, pid, tag):
        from mpi4py import MPI

        return self._sendrecv(mv, MPI.COMM_WORLD.Send_init, pid, tag)

    def recv_pack(self, mv, pid, tag):
        from mpi4py import MPI

        return self._sendrecv(mv, MPI.COMM_WORLD.Recv_init, pid, tag)

    def unpack(self, mv):
        pass
