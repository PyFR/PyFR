# -*- coding: utf-8 -*-

from pyfr.backends.base import ComputeKernel, MPIKernel, NullComputeKernel
from pyfr.backends.openmp.provider import OpenMPKernelProvider
from pyfr.backends.openmp.types import OpenMPXchgMatrix, OpenMPXchgView
from pyfr.nputil import npdtype_to_ctype


class OpenMPPackingKernels(OpenMPKernelProvider):
    def _sendrecv(self, mv, mpipreqfn, pid, tag):
        # If we are an exchange view then extract the exchange matrix
        xchgmat = mv.xchgmat if isinstance(mv, OpenMPXchgView) else mv

        # Create a persistent MPI request to send/recv the pack
        preq = mpipreqfn(xchgmat.data, pid, tag)

        class SendRecvPackKernel(MPIKernel):
            def run(self, queue):
                # Start the request and append us to the list of requests
                preq.Start()
                queue.mpi_reqs.append(preq)

        return SendRecvPackKernel()

    def pack(self, mv):
        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        tpl = self.backend.lookup.get_template('pack')
        src = tpl.render(dtype=npdtype_to_ctype(m.dtype))

        # Build
        kern = self._build_kernel('pack_view', src, 'iiiPPPPP')

        class PackXchgViewKernel(ComputeKernel):
            def run(self, queue):
                kern(v.n, v.nvrow, v.nvcol, v.basedata, v.mapping,
                     v.cstrides or 0, v.rstrides or 0, m)

        return PackXchgViewKernel()

    def send_pack(self, mv, pid, tag):
        from mpi4py import MPI

        return self._sendrecv(mv, MPI.COMM_WORLD.Send_init, pid, tag)

    def recv_pack(self, mv, pid, tag):
        from mpi4py import MPI

        return self._sendrecv(mv, MPI.COMM_WORLD.Recv_init, pid, tag)

    def unpack(self, mv):
        # No-op
        return NullComputeKernel()
