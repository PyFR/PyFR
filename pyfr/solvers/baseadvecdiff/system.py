# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionSystem


class BaseAdvectionDiffusionSystem(BaseAdvectionSystem):
    def _gen_kernels(self):
        super(BaseAdvectionDiffusionSystem, self)._gen_kernels()

        eles = self._eles
        int_inters = self._int_inters
        mpi_inters = self._mpi_inters
        bc_inters = self._bc_inters

        # Element-local kernels
        self._tgradpcoru_upts_kerns = eles.get_tgradpcoru_upts_kern()
        self._tgradcoru_upts_kerns = eles.get_tgradcoru_upts_kern()
        self._tgradcoru_fpts_kerns = eles.get_tgradcoru_fpts_kern()
        self._gradcoru_fpts_kerns = eles.get_gradcoru_fpts_kern()

        self._mpi_inters_vect_fpts0_pack_kerns = \
            mpi_inters.get_vect_fpts0_pack_kern()
        self._mpi_inters_vect_fpts0_send_kerns = \
            mpi_inters.get_vect_fpts0_send_pack_kern()
        self._mpi_inters_vect_fpts0_recv_kerns = \
            mpi_inters.get_vect_fpts0_recv_pack_kern()
        self._mpi_inters_vect_fpts0_unpack_kerns = \
            mpi_inters.get_vect_fpts0_unpack_kern()

        self._int_inters_conu_fpts_kerns = int_inters.get_conu_fpts_kern()
        self._mpi_inters_conu_fpts_kerns = mpi_inters.get_conu_fpts_kern()
        self._bc_inters_conu_fpts_kerns = bc_inters.get_conu_fpts_kern()

    def _get_negdivf(self):
        runall = self._backend.runall
        q1, q2 = self._queues

        q1 << self._disu_fpts_kerns()
        q1 << self._mpi_inters_scal_fpts0_pack_kerns()
        runall([q1])

        q1 << self._int_inters_conu_fpts_kerns()
        q1 << self._bc_inters_conu_fpts_kerns()
        q1 << self._tgradpcoru_upts_kerns()

        q2 << self._mpi_inters_scal_fpts0_send_kerns()
        q2 << self._mpi_inters_scal_fpts0_recv_kerns()
        q2 << self._mpi_inters_scal_fpts0_unpack_kerns()

        runall([q1, q2])

        q1 << self._mpi_inters_conu_fpts_kerns()
        q1 << self._tgradcoru_upts_kerns()
        q1 << self._tgradcoru_fpts_kerns()
        q1 << self._gradcoru_fpts_kerns()
        q1 << self._mpi_inters_vect_fpts0_pack_kerns()
        runall([q1])

        q1 << self._tdisf_upts_kerns()
        q1 << self._tdivtpcorf_upts_kerns()
        q1 << self._int_inters_comm_flux_kerns()
        q1 << self._bc_inters_comm_flux_kerns()

        q2 << self._mpi_inters_vect_fpts0_send_kerns()
        q2 << self._mpi_inters_vect_fpts0_recv_kerns()
        q2 << self._mpi_inters_vect_fpts0_unpack_kerns()

        runall([q1, q2])

        q1 << self._mpi_inters_comm_flux_kerns()
        q1 << self._tdivtconf_upts_kerns()
        q1 << self._negdivconf_upts_kerns()
        runall([q1])
