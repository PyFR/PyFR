# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseSystem


class BaseAdvectionSystem(BaseSystem):
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
        if 'eles/qptsu' in kernels:
            q.enqueue(kernels['eles/qptsu'])
        q.enqueue(kernels['eles/tdisf_curved'])
        q.enqueue(kernels['eles/tdisf_linear'])
        q.enqueue(kernels['eles/tdivtpcorf'])
        q.enqueue(kernels['iint/comm_flux'])
        q.enqueue(kernels['bcint/comm_flux'], t=t)
        q.run(mpireqs['scal_fpts_send_recv'])

        q.enqueue(kernels['mpiint/scal_fpts_unpack'])
        q.enqueue(kernels['mpiint/comm_flux'])
        q.enqueue(kernels['eles/tdivtconf'])
        q.enqueue(kernels['eles/negdivconf'], t=t)
        q.run()

    def preproc(self, t, uinbank):
        q = self._queue
        kernels = self._get_kernels(uinbank, None)
        mpireqs = self._mpireqs

        for b in self._bc_inters:
            b.prepare(t)
        
        # If entropy filtering, compute entropy bounds
        if 'eles/local_entropy' in kernels:
            q.enqueue(kernels['eles/local_entropy'])
            q.enqueue(kernels['mpiint/ent_fpts_pack'])
            q.run()

            q.enqueue(kernels['iint/comm_entropy'])
            q.enqueue(kernels['bcint/comm_entropy'], t=t)
            q.run(mpireqs['ent_fpts_send_recv'])

            q.enqueue(kernels['mpiint/ent_fpts_unpack'])
            q.enqueue(kernels['mpiint/comm_entropy'])
            q.run()

            q.enqueue(kernels['eles/min_entropy'])
            q.run()

    def postproc(self, uinbank):
        q = self._queue
        kernels = self._get_kernels(uinbank, None)

        if 'eles/filter_solution' in kernels:
            q.enqueue(kernels['eles/filter_solution'])
            q.run()


