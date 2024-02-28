from itertools import zip_longest

from pyfr.solvers.base import BaseSystem
from pyfr.util import memoize


class BaseAdvectionSystem(BaseSystem):
    @memoize
    def _rhs_graphs(self, uinbank, foutbank):
        m = self._mpireqs
        k, _ = self._get_kernels(uinbank, foutbank)

        def deps(dk, *names): return self._kdeps(k, dk, *names)

        g1 = self.backend.graph()
        g1.add_mpi_reqs(m['scal_fpts_recv'] + m['ent_fpts_recv'])

        # Perform post-processing of the previous solution stage
        g1.add_all(k['eles/entropy_filter'])

        # Interpolate the solution to the flux points
        for l in k['eles/disu']:
            g1.add(l, deps=deps(l, 'eles/entropy_filter'))

        # Pack and send these interpolated solutions to our neighbours
        g1.add_all(k['mpiint/scal_fpts_pack'], deps=k['eles/disu'])
        for send, pack in zip(m['scal_fpts_send'], k['mpiint/scal_fpts_pack']):
            g1.add_mpi_req(send, deps=[pack])

        # If entropy filtering, pack and send the entropy values to neighbours
        g1.add_all(k['mpiint/ent_fpts_pack'], deps=k['eles/entropy_filter'])
        for send, pack in zip(m['ent_fpts_send'], k['mpiint/ent_fpts_pack']):
            g1.add_mpi_req(send, deps=[pack])

        # Compute common entropy minima at internal/boundary interfaces
        g1.add_all(k['iint/comm_entropy'],
                   deps=k['eles/entropy_filter'] + k['mpiint/ent_fpts_pack'])
        g1.add_all(k['bcint/comm_entropy'],
                   deps=k['eles/disu'])

        # Compute the common normal flux at our internal/boundary interfaces
        g1.add_all(k['iint/comm_flux'],
                   deps=k['eles/disu'] + k['mpiint/scal_fpts_pack'])
        g1.add_all(k['bcint/comm_flux'],
                   deps=k['eles/disu'] + k['bcint/comm_entropy'])

        # Make a copy of the solution (if used by source terms)
        g1.add_all(k['eles/copy_soln'], deps=k['eles/entropy_filter'])

        # Interpolate the solution to the quadrature points
        g1.add_all(k['eles/qptsu'], deps=k['eles/entropy_filter'])

        # Compute the transformed flux
        for l in k['eles/tdisf']:
            ldeps = deps(l, 'eles/qptsu')
            g1.add(l, deps=ldeps + k['eles/entropy_filter'])

        # Compute the transformed divergence of the partially corrected flux
        for l in k['eles/tdivtpcorf']:
            ldeps = deps(l, 'eles/tdisf', 'eles/copy_soln', 'eles/disu')
            g1.add(l, deps=ldeps + k['mpiint/scal_fpts_pack'])

        # Group tdisf, qptsu, and tdivtpcorf kernels
        kgroup = [k['eles/qptsu'], k['eles/tdisf'], k['eles/tdivtpcorf']]
        for k1, k2, k3 in zip_longest(*kgroup):
            self._group(g1, [k1, k2, k3], subs=[[(k1, 'out'), (k2, 'u')],
                                                [(k2, 'f'), (k3, 'b')]])

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

        # Group tdivtconf and negdivconf kernels
        for k1, k2 in zip_longest(k['eles/tdivtconf'], k['eles/negdivconf']):
            self._group(g2, [k1, k2])

        g2.commit()

        return g1, g2

    @memoize
    def _preproc_graphs(self, uinbank):
        m = self._mpireqs
        k, _ = self._get_kernels(uinbank, None)

        def deps(dk, *names): return self._kdeps(k, dk, *names)

        g1 = self.backend.graph()
        g1.add_mpi_reqs(m['ent_fpts_recv'])

        # Interpolate the solution to the flux points
        if 'eles/local_entropy' in k:
            g1.add_all(k['eles/disu'])

        # Compute local minimum entropy within element
        g1.add_all(k['eles/local_entropy'])

        # Pack and send the entropy values to neighbours
        g1.add_all(k['mpiint/ent_fpts_pack'], deps=k['eles/local_entropy'])
        for send, pack in zip(m['ent_fpts_send'], k['mpiint/ent_fpts_pack']):
            g1.add_mpi_req(send, deps=[pack])

        # Compute common entropy minima at internal/boundary interfaces
        g1.add_all(k['iint/comm_entropy'], deps=k['eles/local_entropy'])
        g1.add_all(k['bcint/comm_entropy'],
                   deps=k['eles/local_entropy'] + k['eles/disu'])
        g1.commit()

        if 'mpiint/comm_entropy' in k:
            # Compute common entropy minima at MPI interfaces
            g2 = self.backend.graph()

            g2.add_all(k['mpiint/ent_fpts_unpack'])
            for l in k['mpiint/comm_entropy']:
                g2.add(l, deps=deps(l, 'mpiint/ent_fpts_unpack'))
            g2.commit()

            return g1, g2
        else:
            return g1,

    def postproc(self, uinbank):
        k, _ = self._get_kernels(uinbank, None)

        self.backend.run_kernels(k['eles/entropy_filter'])
