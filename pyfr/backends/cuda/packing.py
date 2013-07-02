# -*- coding: utf-8 -*-

from mpi4py import MPI

import pycuda.driver as cuda

from pyfr.backends.base import ComputeKernel, MPIKernel
from pyfr.backends.cuda.provider import CUDAKernelProvider
from pyfr.backends.cuda.types import CUDAMPIMatrix, CUDAMPIView

from pyfr.nputil import npdtype_to_ctype


class CUDAPackingKernels(CUDAKernelProvider):
    def _packmodopts(self, mpiview):
        return dict(dtype=npdtype_to_ctype(mpiview.mpimat.dtype),
                    vlen=mpiview.view.vlen)

    def _packunpack_mpimat(self, op, mpimat):
        class PackUnpackKernel(ComputeKernel):
            if op == 'pack':
                def run(self, scomp, scopy):
                    cuda.memcpy_dtoh_async(mpimat.hdata, mpimat.data, scomp)
            else:
                def run(self, scomp, scopy):
                    cuda.memcpy_htod_async(mpimat.data, mpimat.hdata, scomp)

        return PackUnpackKernel()

    def _packunpack_mpiview(self, op, mpiview):
        # An MPI view is simply a regular view plus an MPI matrix
        v, m = mpiview.view, mpiview.mpimat

        # Get the CUDA pack/unpack kernel from the pack module
        fn = self._get_function('pack', op + '_view', 'iiPPPiii',
                                tplparams=self._packmodopts(mpiview))

        # Compute the grid and thread-block size
        grid, block = self._get_2d_grid_block(fn, v.nrow, v.ncol)

        # Create a CUDA event
        event = cuda.Event(cuda.event_flags.DISABLE_TIMING)

        class ViewPackUnpackKernel(ComputeKernel):
            def run(self, scomp, scopy):
                # If we are unpacking then copy the host buffer to the GPU
                if op == 'unpack':
                    cuda.memcpy_htod_async(m.data, m.hdata, scopy)
                    event.record(scopy)
                    scomp.wait_for_event(event)

                # Call the CUDA kernel (pack or unpack)
                fn.prepared_async_call(grid, block, scomp, v.nrow, v.ncol,
                                       v.mapping, v.strides, m,
                                       v.mapping.leaddim, v.strides.leaddim,
                                       m.leaddim)

                # If we have been packing then copy the GPU buffer to the host
                if op == 'pack':
                    event.record(scomp)
                    scopy.wait_for_event(event)
                    cuda.memcpy_dtoh_async(m.hdata, m.data, scopy)

        return ViewPackUnpackKernel()

    def _packunpack(self, op, mv):
        if isinstance(mv, CUDAMPIMatrix):
            return self._packunpack_mpimat(op, mv)
        elif isinstance(mv, CUDAMPIView):
            return self._packunpack_mpiview(op, mv)
        else:
            raise TypeError('Can only pack MPI views and MPI matrices')

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
        return self._packunpack('pack', mv)

    def send_pack(self, mv, pid, tag):
        return self._sendrecv(mv, MPI.COMM_WORLD.Send_init, pid, tag)

    def recv_pack(self, mv, pid, tag):
        return self._sendrecv(mv, MPI.COMM_WORLD.Recv_init, pid, tag)

    def unpack(self, mv):
        return self._packunpack('unpack', mv)
