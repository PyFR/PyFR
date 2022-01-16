# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseSystem


class BaseAdvectionSystem(BaseSystem):
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
        k += kernels['eles/qptsu']
        k += kernels['eles/tdisf_curved']
        k += kernels['eles/tdisf_linear']
        k += kernels['eles/tdivtpcorf']
        k += kernels['iint/comm_flux']
        k += kernels['bcint/comm_flux']
        run(k, mpireqs['scal_fpts_send_recv'])

        k = []
        k += kernels['mpiint/scal_fpts_unpack']
        k += kernels['mpiint/comm_flux']
        k += kernels['eles/tdivtconf']
        k += kernels['eles/negdivconf']
        run(k)
