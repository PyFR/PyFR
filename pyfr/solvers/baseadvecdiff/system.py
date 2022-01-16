# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionSystem


class BaseAdvectionDiffusionSystem(BaseAdvectionSystem):
    def rhs(self, t, uinbank, foutbank):
        run = self.backend.run
        kernels = self._prepare_kernels(t, uinbank, foutbank)
        mpireqs = self._mpireqs

        k = []
        k += kernels['eles/disu']
        k += kernels['mpiint/scal_fpts_pack']
        run(k)

        k = []
        k += kernels['eles/copy_soln']
        k += kernels['iint/copy_fpts']
        k += kernels['iint/con_u']
        k += kernels['bcint/con_u']
        k += kernels['eles/shocksensor']
        k += kernels['mpiint/artvisc_fpts_pack']
        k += kernels['eles/tgradpcoru_upts']
        run(k, mpireqs['scal_fpts_send_recv'])

        k = []
        k += kernels['mpiint/scal_fpts_unpack']
        k += kernels['mpiint/con_u']
        k += kernels['eles/tgradcoru_upts']
        k += kernels['eles/gradcoru_upts_curved']
        k += kernels['eles/gradcoru_upts_linear']
        k += kernels['eles/gradcoru_fpts']
        k += kernels['mpiint/vect_fpts_pack']
        run(k, mpireqs['artvisc_fpts_send_recv'])

        k = []
        k += kernels['mpiint/artvisc_fpts_unpack']
        k += kernels['eles/gradcoru_qpts']
        k += kernels['eles/qptsu']
        k += kernels['eles/tdisf_curved']
        k += kernels['eles/tdisf_linear']
        k += kernels['eles/tdivtpcorf']
        k += kernels['iint/comm_flux']
        k += kernels['bcint/comm_flux']
        run(k, mpireqs['vect_fpts_send_recv'])

        k = []
        k += kernels['mpiint/vect_fpts_unpack']
        k += kernels['mpiint/comm_flux']
        k += kernels['eles/tdivtconf']
        k += kernels['eles/negdivconf']
        run(k)

    def compute_grads(self, t, uinbank):
        run = self.backend.run
        kernels = self._prepare_kernels(t, uinbank, None)
        mpireqs = self._mpireqs

        k = []
        k += kernels['eles/disu']
        k += kernels['mpiint/scal_fpts_pack']
        run(k)

        k = []
        k += kernels['iint/copy_fpts']
        k += kernels['iint/con_u']
        k += kernels['bcint/con_u']
        k += kernels['eles/tgradpcoru_upts']
        run(k, mpireqs['scal_fpts_send_recv'])

        k += kernels['mpiint/scal_fpts_unpack']
        k += kernels['mpiint/con_u']
        k += kernels['eles/tgradcoru_upts']
        k += kernels['eles/gradcoru_upts_curved']
        k += kernels['eles/gradcoru_upts_linear']
        run(k)
