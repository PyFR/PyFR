# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionSystem


class BaseAdvectionDiffusionSystem(BaseAdvectionSystem):
    def rhs(self, t, uinbank, foutbank):
        q = self._queue
        kernels = self._get_kernels(uinbank, foutbank)
        mpireqs = self._mpireqs

        for b in self._bc_inters:
            b.prepare(t)

        q.enqueue(kernels['eles/disu'])
        q.enqueue(kernels['mpiint/scal_fpts_pack'])
        q.run()

        if 'eles/copy_soln' in kernels:
            q.enqueue(kernels['eles/copy_soln'])
        if 'iint/copy_fpts' in kernels:
            q.enqueue(kernels['iint/copy_fpts'])
        q.enqueue(kernels['iint/con_u'])
        q.enqueue(kernels['bcint/con_u'], t=t)
        if 'eles/shocksensor' in kernels:
            q.enqueue(kernels['eles/shocksensor'])
            q.enqueue(kernels['mpiint/artvisc_fpts_pack'])
        q.enqueue(kernels['eles/tgradpcoru_upts'])
        q.run(mpireqs['scal_fpts_send_recv'])

        q.enqueue(kernels['mpiint/scal_fpts_unpack'])
        q.enqueue(kernels['mpiint/con_u'])
        q.enqueue(kernels['eles/tgradcoru_upts'])
        q.enqueue(kernels['eles/gradcoru_upts_curved'])
        q.enqueue(kernels['eles/gradcoru_upts_linear'])
        q.enqueue(kernels['eles/gradcoru_fpts'])
        q.enqueue(kernels['mpiint/vect_fpts_pack'])
        q.run(mpireqs['artvisc_fpts_send_recv'])

        if 'eles/shockvar' in kernels:
            q.enqueue(kernels['mpiint/artvisc_fpts_unpack'])
        if 'eles/gradcoru_qpts' in kernels:
            q.enqueue(kernels['eles/gradcoru_qpts'])
            q.enqueue(kernels['eles/qptsu'])
        q.enqueue(kernels['eles/tdisf_curved'])
        q.enqueue(kernels['eles/tdisf_linear'])
        q.enqueue(kernels['eles/tdivtpcorf'])
        q.enqueue(kernels['iint/comm_flux'])
        q.enqueue(kernels['bcint/comm_flux'], t=t)
        q.run(mpireqs['vect_fpts_send_recv'])

        q.enqueue(kernels['mpiint/vect_fpts_unpack'])
        q.enqueue(kernels['mpiint/comm_flux'])
        q.enqueue(kernels['eles/tdivtconf'])
        q.enqueue(kernels['eles/negdivconf'], t=t)
        q.run()

    def compute_grads(self, t, uinbank):
        q = self._queue
        kernels = self._get_kernels(uinbank, None)
        mpireqs = self._mpireqs

        for b in self._bc_inters:
            b.prepare(t)

        q.enqueue(kernels['eles/disu'])
        q.enqueue(kernels['mpiint/scal_fpts_pack'])
        q.run()

        if 'iint/copy_fpts' in kernels:
            q.enqueue(kernels['iint/copy_fpts'])
        q.enqueue(kernels['iint/con_u'])
        q.enqueue(kernels['bcint/con_u'], t=t)
        q.enqueue(kernels['eles/tgradpcoru_upts'])
        q.run(mpireqs['scal_fpts_send_recv'])

        q.enqueue(kernels['mpiint/scal_fpts_unpack'])
        q.enqueue(kernels['mpiint/con_u'])
        q.enqueue(kernels['eles/tgradcoru_upts'])
        q.enqueue(kernels['eles/gradcoru_upts_curved'])
        q.enqueue(kernels['eles/gradcoru_upts_linear'])
        q.run()
