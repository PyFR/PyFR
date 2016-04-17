# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionSystem
from pyfr.solvers.navstokes.elements import NavierStokesElements
from pyfr.solvers.navstokes.inters import (NavierStokesBaseBCInters,
                                           NavierStokesIntInters,
                                           NavierStokesMPIInters)


class NavierStokesSystem(BaseAdvectionDiffusionSystem):
    name = 'navier-stokes'

    elementscls = NavierStokesElements
    intinterscls = NavierStokesIntInters
    mpiinterscls = NavierStokesMPIInters
    bbcinterscls = NavierStokesBaseBCInters

    def rhs(self, t, uinbank, foutbank):
        runall = self.backend.runall
        q1, q2 = self._queues
        kernels = self._kernels

        self.eles_scal_upts_inb.active = uinbank
        self.eles_scal_upts_outb.active = foutbank

        q1 << kernels['eles', 'disu']()
        q1 << kernels['mpiint', 'scal_fpts_pack']()
        runall([q1])

        if ('eles', 'copy_soln') in kernels:
            q1 << kernels['eles', 'copy_soln']()
        q1 << kernels['iint', 'con_u']()
        q1 << kernels['bcint', 'con_u'](t=t)
        q1 << kernels['eles', 'tgradpcoru_upts']()
        if ('eles', 'art_visc') in kernels:
            q1 << kernels['eles', 'entropy']()
            q1 << kernels['eles', 'modal_entropy']()
            q1 << kernels['eles', 'art_visc']()
            q1 << kernels['mpiint', 'avis_fpts_pack']()

        q2 << kernels['mpiint', 'scal_fpts_send']()
        q2 << kernels['mpiint', 'scal_fpts_recv']()
        q2 << kernels['mpiint', 'scal_fpts_unpack']()

        runall([q1, q2])

        q1 << kernels['mpiint', 'con_u']()
        q1 << kernels['eles', 'tgradcoru_upts']()
        q1 << kernels['eles', 'gradcoru_upts']()
        q1 << kernels['eles', 'gradcoru_fpts']()
        q1 << kernels['mpiint', 'vect_fpts_pack']()
        if ('eles', 'avis') in kernels:
            q2 << kernels['mpiint', 'avis_fpts_send']()
            q2 << kernels['mpiint', 'avis_fpts_recv']()
            q2 << kernels['mpiint', 'avis_fpts_unpack']()

        runall([q1, q2])

        if ('eles', 'gradcoru_qpts') in kernels:
            q1 << kernels['eles', 'gradcoru_qpts']()
        q1 << kernels['eles', 'tdisf']()
        q1 << kernels['eles', 'tdivtpcorf']()
        q1 << kernels['iint', 'comm_flux']()
        q1 << kernels['bcint', 'comm_flux'](t=t)

        q2 << kernels['mpiint', 'vect_fpts_send']()
        q2 << kernels['mpiint', 'vect_fpts_recv']()
        q2 << kernels['mpiint', 'vect_fpts_unpack']()

        runall([q1, q2])

        q1 << kernels['mpiint', 'comm_flux']()
        q1 << kernels['eles', 'tdivtconf']()
        if ('eles', 'tdivf_qpts') in kernels:
            q1 << kernels['eles', 'tdivf_qpts']()
            q1 << kernels['eles', 'negdivconf'](t=t)
            q1 << kernels['eles', 'divf_upts']()
        else:
            q1 << kernels['eles', 'negdivconf'](t=t)
        runall([q1])
