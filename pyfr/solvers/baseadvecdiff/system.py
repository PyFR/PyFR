from itertools import zip_longest

from pyfr.solvers.baseadvec import BaseAdvectionSystem
from pyfr.util import memoize


class BaseAdvectionDiffusionSystem(BaseAdvectionSystem):
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
        g1.add_all(k['eles/disu'], deps=k['eles/entropy_filter'])

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

        # Make a copy of the solution (if used by source terms)
        g1.add_all(k['eles/copy_soln'], deps=k['eles/entropy_filter'])

        # Compute the common solution at our internal/boundary interfaces
        for l in k['eles/copy_fpts']:
            g1.add(l, deps=deps(l, 'eles/disu'))
        kdeps = k['eles/copy_fpts'] or k['eles/disu']
        g1.add_all(k['iint/con_u'], deps=kdeps + k['mpiint/scal_fpts_pack'])
        g1.add_all(k['bcint/con_u'], deps=kdeps)

        # Run the shock sensor (if enabled)
        g1.add_all(k['eles/shocksensor'])
        g1.add_all(k['mpiint/artvisc_fpts_pack'], deps=k['eles/shocksensor'])

        # Compute the transformed gradient of the partially corrected solution
        g1.add_all(k['eles/tgradpcoru_upts'],
                   deps=k['mpiint/scal_fpts_pack'] + k['eles/entropy_filter'])
        g1.commit()

        g2 = self.backend.graph()
        g2.add_mpi_reqs(m['artvisc_fpts_send'] + m['artvisc_fpts_recv'])
        g2.add_mpi_reqs(m['vect_fpts_recv'])

        # Compute the common solution at our MPI interfaces
        g2.add_all(k['mpiint/scal_fpts_unpack'])
        for l in k['mpiint/con_u']:
            g2.add(l, deps=deps(l, 'mpiint/scal_fpts_unpack'))

        # Compute common entropy minima at MPI interfaces
        g2.add_all(k['mpiint/ent_fpts_unpack'])
        for l in k['mpiint/comm_entropy']:
            g2.add(l, deps=deps(l, 'mpiint/ent_fpts_unpack'))

        # Compute the transformed gradient of the corrected solution
        g2.add_all(k['eles/tgradcoru_upts'], deps=k['mpiint/con_u'])

        # Obtain the physical gradients at the solution points
        for l in k['eles/gradcoru_upts']:
            g2.add(l, deps=deps(l, 'eles/tgradcoru_upts'))

        # Compute the fused transformed flux and corrected gradient
        for l in k['eles/tdisf_fused']:
            ldeps = deps(l, 'eles/tgradcoru_upts')
            g2.add(l, deps=ldeps)

        # Interpolate these gradients to the flux points
        for l in k['eles/gradcoru_fpts']:
            ldeps = deps(l, 'eles/tdisf_fused', 'eles/gradcoru_upts')
            g2.add(l, deps=ldeps)

        # Set dependencies for interface flux interpolation
        ideps = k['eles/gradcoru_fpts'] or k['eles/gradcoru_upts']

        # Pack and send these interpolated gradients to our neighbours
        g2.add_all(k['mpiint/vect_fpts_pack'], deps=ideps)
        for send, pack in zip(m['vect_fpts_send'], k['mpiint/vect_fpts_pack']):
            g2.add_mpi_req(send, deps=[pack])

        # Compute the common normal flux at our internal/boundary interfaces
        g2.add_all(k['iint/comm_flux'],
                   deps=ideps, pdeps=k['mpiint/vect_fpts_pack'])
        g2.add_all(k['bcint/comm_flux'], deps=ideps)

        # Interpolate the gradients to the quadrature points
        for l in k['eles/gradcoru_qpts']:
            ldeps = deps(l, 'eles/gradcoru_upts')
            g2.add(l, deps=ldeps, pdeps=k['mpiint/vect_fpts_pack'])

        # Interpolate the solution to the quadrature points
        g2.add_all(k['eles/qptsu'])

        # Compute the transformed flux
        for l in k['eles/tdisf']:
            if k['eles/qptsu']:
                ldeps = deps(l, 'eles/gradcoru_qpts', 'eles/qptsu')
            elif k['eles/gradcoru_fpts']:
                ldeps = deps(l, 'eles/gradcoru_fpts')
            else:
                ldeps = deps(l, 'eles/gradcoru_upts')
            g2.add(l, deps=ldeps)

        # Compute the transformed divergence of the partially corrected flux
        for l in k['eles/tdivtpcorf']:
            g2.add(l, deps=deps(l, 'eles/tdisf', 'eles/tdisf_fused'))

        kgroup = [
            k['eles/tgradcoru_upts'], k['eles/gradcoru_upts'],
            k['eles/tdisf_fused'], k['eles/gradcoru_fpts'],
            k['eles/gradcoru_qpts'], k['eles/qptsu'],
            k['eles/tdisf'], k['eles/tdivtpcorf']
        ]
        for ks in zip_longest(*kgroup):
            self._group(g2, ks, subs=[
                [(ks[5], 'out'), (ks[6], 'u')],
                [(ks[2], 'f'), (ks[4], 'out'), (ks[6], 'f'), (ks[7], 'b')]
            ])

        g2.commit()

        g3 = self.backend.graph()

        # Compute the common normal flux at our MPI interfaces
        g3.add_all(k['mpiint/artvisc_fpts_unpack'])
        g3.add_all(k['mpiint/vect_fpts_unpack'])
        for l in k['mpiint/comm_flux']:
            ldeps = deps(l, 'mpiint/artvisc_fpts_unpack',
                         'mpiint/vect_fpts_unpack')
            g3.add(l, deps=ldeps)

        # Compute the transformed divergence of the corrected flux
        g3.add_all(k['eles/tdivtconf'], deps=k['mpiint/comm_flux'])

        # Obtain the physical divergence of the corrected flux
        for l in k['eles/negdivconf']:
            g3.add(l, deps=deps(l, 'eles/tdivtconf'))

        # Group tdivtconf and negdivconf kernels
        for k1, k2 in zip_longest(k['eles/tdivtconf'], k['eles/negdivconf']):
            self._group(g3, [k1, k2])

        g3.commit()

        return g1, g2, g3

    @memoize
    def _compute_grads_graph(self, uinbank):
        m = self._mpireqs
        k, _ = self._get_kernels(uinbank, None)

        def deps(dk, *names): return self._kdeps(k, dk, *names)

        g1 = self.backend.graph()
        g1.add_mpi_reqs(m['scal_fpts_recv'])

        # Interpolate the solution to the flux points
        g1.add_all(k['eles/disu'])

        # Pack and send these interpolated solutions to our neighbours
        g1.add_all(k['mpiint/scal_fpts_pack'], deps=k['eles/disu'])
        for send, pack in zip(m['scal_fpts_send'], k['mpiint/scal_fpts_pack']):
            g1.add_mpi_req(send, deps=[pack])

        # Compute the common solution at our internal/boundary interfaces
        for l in k['eles/copy_fpts']:
            g1.add(l, deps=deps(l, 'eles/disu'))
        kdeps = k['eles/copy_fpts'] or k['eles/disu']
        g1.add_all(k['iint/con_u'], deps=kdeps)
        g1.add_all(k['bcint/con_u'], deps=kdeps)

        # Compute the transformed gradient of the partially corrected solution
        g1.add_all(k['eles/tgradpcoru_upts'])
        g1.commit()

        g2 = self.backend.graph()

        # Compute the common solution at our MPI interfaces
        g2.add_all(k['mpiint/scal_fpts_unpack'])
        for l in k['mpiint/con_u']:
            g2.add(l, deps=deps(l, 'mpiint/scal_fpts_unpack'))

        # Compute the transformed gradient of the corrected solution
        g2.add_all(k['eles/tgradcoru_upts'], deps=k['mpiint/con_u'])

        # Obtain the physical gradients at the solution points
        for l in k['eles/gradcoru_u']:
            g2.add(l, deps=deps(l, 'eles/tgradcoru_upts'))
        g2.commit()

        return g1, g2
