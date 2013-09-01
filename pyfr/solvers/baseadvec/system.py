# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseSystem


class BaseAdvectionSystem(BaseSystem):
    _nqueues = 2

    def _gen_kernels(self):
        eles = self._eles
        int_inters = self._int_inters
        mpi_inters = self._mpi_inters
        bc_inters = self._bc_inters

        # Generate the kernels over each element type
        self._disu_fpts_kerns = eles.get_disu_fpts_kern()
        self._tdisf_upts_kerns = eles.get_tdisf_upts_kern()
        self._tdivtpcorf_upts_kerns = eles.get_tdivtpcorf_upts_kern()
        self._tdivtconf_upts_kerns = eles.get_tdivtconf_upts_kern()
        self._negdivconf_upts_kerns = eles.get_negdivconf_upts_kern()

        # Generate MPI sending/recving kernels over each MPI interface
        self._mpi_inters_scal_fpts0_pack_kerns = \
            mpi_inters.get_scal_fpts0_pack_kern()
        self._mpi_inters_scal_fpts0_send_kerns = \
            mpi_inters.get_scal_fpts0_send_pack_kern()
        self._mpi_inters_scal_fpts0_recv_kerns = \
            mpi_inters.get_scal_fpts0_recv_pack_kern()
        self._mpi_inters_scal_fpts0_unpack_kerns = \
            mpi_inters.get_scal_fpts0_unpack_kern()

        # Generate the Riemann solvers for the various interface types
        self._int_inters_comm_flux_kerns = int_inters.get_comm_flux_kern()
        self._mpi_inters_comm_flux_kerns = mpi_inters.get_comm_flux_kern()
        self._bc_inters_comm_flux_kerns = bc_inters.get_comm_flux_kern()

    def _get_negdivf(self):
        runall = self._backend.runall
        q1, q2 = self._queues

        # Evaluate the solution at the flux points and pack up any
        # flux point solutions which are on our side of an MPI
        # interface
        q1 << self._disu_fpts_kerns()
        q1 << self._mpi_inters_scal_fpts0_pack_kerns()
        runall([q1])

        # Evaluate the flux at each of the solution points and take the
        # divergence of this to yield the transformed, partially
        # corrected, flux divergence.  Finally, solve the Riemann
        # problem at each interface to yield a common flux
        q1 << self._tdisf_upts_kerns()
        q1 << self._tdivtpcorf_upts_kerns()
        q1 << self._int_inters_comm_flux_kerns()
        q1 << self._bc_inters_comm_flux_kerns()

        # Send the MPI interface buffers we have just packed and
        # receive the corresponding buffers from our peers.  Then
        # proceed to unpack these received buffers
        q2 << self._mpi_inters_scal_fpts0_send_kerns()
        q2 << self._mpi_inters_scal_fpts0_recv_kerns()
        q2 << self._mpi_inters_scal_fpts0_unpack_kerns()

        runall([q1, q2])

        # Solve the remaining Riemann problems for the MPI interfaces
        # and use the complete set of common fluxes to generate the
        # fully corrected transformed flux divergence.  Finally,
        # negate and un-transform this divergence to give -∇·f.
        q1 << self._mpi_inters_comm_flux_kerns()
        q1 << self._tdivtconf_upts_kerns()
        q1 << self._negdivconf_upts_kerns()
        runall([q1])
