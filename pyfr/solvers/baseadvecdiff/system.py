# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionSystem


class BaseAdvectionDiffusionSystem(BaseAdvectionSystem):
    def _get_negdivf(self):
        runall = self._backend.runall
        q1, q2 = self._queues
        kernels = self._kernels

        q1 << kernels['eles', 'disu_fpts']()
        q1 << kernels['mpiint', 'scal_fpts_pack']()
        runall([q1])

        q1 << kernels['iint', 'con_u']()
        q1 << kernels['bcint', 'con_u']()
        q1 << kernels['eles', 'tgradpcoru_upts']()

        q2 << kernels['mpiint', 'scal_fpts_send']()
        q2 << kernels['mpiint', 'scal_fpts_recv']()
        q2 << kernels['mpiint', 'scal_fpts_unpack']()

        runall([q1, q2])

        q1 << kernels['mpiint', 'con_u']()
        q1 << kernels['eles', 'tgradcoru_upts']()
        q1 << kernels['eles', 'gradcoru_upts']()
        q1 << kernels['eles', 'gradcoru_fpts']()
        q1 << kernels['mpiint', 'vect_fpts_pack']()
        runall([q1])

        q1 << kernels['eles', 'tdisf_upts']()
        q1 << kernels['eles', 'tdivtpcorf_upts']()
        q1 << kernels['iint', 'comm_flux']()
        q1 << kernels['bcint', 'comm_flux']()

        q2 << kernels['mpiint', 'vect_fpts_send']()
        q2 << kernels['mpiint', 'vect_fpts_recv']()
        q2 << kernels['mpiint', 'vect_fpts_unpack']()

        runall([q1, q2])

        q1 << kernels['mpiint', 'comm_flux']()
        q1 << kernels['eles', 'tdivtconf_upts']()
        q1 << kernels['eles', 'negdivconf_upts']()
        runall([q1])
