# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseSystem
from pyfr.util import memoize


class BaseAdvectionSystem(BaseSystem):
    @memoize
    def _rhs_graphs(self, uinbank, foutbank):
        m = self._mpireqs
        k, _ = self._get_kernels(uinbank, foutbank)

        def deps(dk, *names): return self._kdeps(k, dk, *names)

        g1 = self.backend.graph()
        g1.add_mpi_reqs(m['scal_fpts_recv'])

        # Perform post-processing of the previous solution stage
        g1.add_all(k['eles/filter_solution'])

        # Interpolate the solution to the flux points
        g1.add_all(k['eles/disu'], deps=k['eles/filter_solution'])

        # Pack and send these interpolated solutions to our neighbours
        g1.add_all(k['mpiint/scal_fpts_pack'], deps=k['eles/disu'])
        for send, pack in zip(m['scal_fpts_send'], k['mpiint/scal_fpts_pack']):
            g1.add_mpi_req(send, deps=[pack])

        # If entropy filtering, compute entropy bounds
        g1.add_mpi_reqs(m['ent_fpts_recv'])

        # Pack and send the entropy values to neighbors
        g1.add_all(k['mpiint/ent_fpts_pack'], deps=k['eles/filter_solution'])
        for send, pack in zip(m['ent_fpts_send'], k['mpiint/ent_fpts_pack']):
            g1.add_mpi_req(send, deps=[pack])

        # Compute common entropy minima at internal/boundary interfaces
        g1.add_all(k['iint/comm_entropy'],
                   deps=k['eles/filter_solution'] + k['mpiint/ent_fpts_pack'])
        g1.add_all(k['bcint/comm_entropy'],
                   deps=k['eles/filter_solution'] + k['eles/disu'])

        # Compute the common normal flux at our internal/boundary interfaces
        g1.add_all(k['iint/comm_flux'],
                   deps=k['eles/disu'] + k['mpiint/scal_fpts_pack'])
        g1.add_all(k['bcint/comm_flux'],
                   deps=k['eles/disu'] + k['bcint/comm_entropy'])

        # Make a copy of the solution (if used by source terms)
        g1.add_all(k['eles/copy_soln'], deps=k['eles/filter_solution'])

        # Interpolate the solution to the quadrature points
        g1.add_all(k['eles/qptsu'], deps=k['eles/filter_solution'])

        # Compute the transformed flux
        for l in k['eles/tdisf_curved'] + k['eles/tdisf_linear']:
            ldeps = deps(l, 'eles/qptsu')
            g1.add(l, deps=ldeps + k['eles/filter_solution'])

        # Compute the transformed divergence of the partially corrected flux
        for l in k['eles/tdivtpcorf']:
            ldeps = deps(l, 'eles/tdisf_curved', 'eles/tdisf_linear',
                         'eles/copy_soln', 'eles/disu')
            g1.add(l, deps=ldeps + k['mpiint/scal_fpts_pack'])
        g1.commit()

        g2 = self.backend.graph()

        # Compute the common normal flux at our MPI interfaces
        g2.add_all(k['mpiint/scal_fpts_unpack'])
        for l in k['mpiint/comm_flux']:
            g2.add(l, deps=deps(l, 'mpiint/scal_fpts_unpack'))

        # Compute common entropy minima at MPI interfaces
        g2.add_all(k['mpiint/ent_fpts_unpack'])
        for l in k['mpiint/comm_entropy']:
            g2.add(l, deps=deps(l, 'mpiint/ent_fpts_unpack'))

        # Compute the transformed divergence of the corrected flux
        g2.add_all(k['eles/tdivtconf'], deps=k['mpiint/comm_flux'])

        # Obtain the physical divergence of the corrected flux
        for l in k['eles/negdivconf']:
            g2.add(l, deps=deps(l, 'eles/tdivtconf'))
        g2.commit()

        return g1, g2

    def _preproc_graphs(self, uinbank):
        m = self._mpireqs
        k, _ = self._get_kernels(uinbank, None)

        def deps(dk, *names): return self._kdeps(k, dk, *names)

        g1 = self.backend.graph()

        # Interpolate the solution to the flux points
        g1.add_all(k['eles/disu'])

        # Compute local minimum entropy within element
        g1.add_all(k['eles/local_entropy'])

        # Pack and send the entropy values to neighbors
        g1.add_all(k['mpiint/ent_fpts_pack'], deps=k['eles/local_entropy'])
        for send, pack in zip(m['ent_fpts_send'], k['mpiint/ent_fpts_pack']):
            g1.add_mpi_req(send, deps=[pack])

        # Compute common entropy minima at internal/boundary interfaces
        g1.add_all(k['iint/comm_entropy'],
                   deps=k['eles/local_entropy'] + k['mpiint/ent_fpts_pack'])
        g1.add_all(k['bcint/comm_entropy'],
                   deps=k['eles/local_entropy'] + k['eles/disu'])
        g1.commit()

        g2 = self.backend.graph()

        # Compute common entropy minima at MPI interfaces
        g2.add_all(k['mpiint/ent_fpts_unpack'])
        for l in k['mpiint/comm_entropy']:
            g2.add(l, deps=deps(l, 'mpiint/ent_fpts_unpack'))
        g2.commit()

        return g1, g2

    def postproc(self, uinbank):
        k, _ = self._get_kernels(uinbank, None)

        self.backend.run_kernels(k['eles/filter_solution'])
