from itertools import zip_longest

from pyfr.cache import memoize
from pyfr.solvers.baseadvec import BaseAdvectionSystem


class BaseAdvectionDiffusionSystem(BaseAdvectionSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Register vect_fpts conditional MPI exchange
        vect_v = [(m, m._vect_lhs, m._vect_rhs) for m in self._mpi_inters]
        self.register_mpi_exchange(
            'vect_fpts', vect_v,
            send=lambda m: m.c['ldg-beta'] != -0.5,
            recv=lambda m: m.c['ldg-beta'] != 0.5,
        )

    @memoize
    def _rhs_graphs(self, uinbank, foutbank):
        m = self._mpireqs
        k, *_ = self._get_kernels(uinbank, foutbank)

        def deps(dk, *names): return self._kdeps(k, dk, *names)

        # Graph: interpolate solution, exchange, compute common solution
        g_soln = self.backend.graph()
        g_soln.add_mpi_reqs(m['scal_fpts_recv'])

        # Interpolate the solution to the flux points
        g_soln.add_all(k['eles/disu'], deps=k['eles/entropy_filter'])

        # Shock capturing kernels which run before the solution exchange
        self.shock_capturing.add_to_graph_pre_recv(g_soln, k, m)

        # Pack and send these interpolated solutions to our neighbours
        g_soln.add_all(k['mpiint/scal_fpts_pack'], deps=k['eles/disu'])
        for send, pack in zip(m['scal_fpts_send'],
                              k['mpiint/scal_fpts_pack']):
            g_soln.add_mpi_req(send, deps=[pack])

        # Compute the common solution at our internal/boundary interfaces
        g_soln.add_all(k['iint/con_u'],
                       deps=k['eles/disu'] + k['mpiint/scal_fpts_pack'])
        g_soln.add_all(k['bcint/con_u'], deps=k['eles/disu'])

        g_soln.commit()

        # Graph: compute gradients, flux, and partial divergence
        g_grad_flux = self.backend.graph()
        g_grad_flux.add_mpi_reqs(m['vect_fpts_recv'])

        # Unpack MPI face data (may be empty when unpack is a no-op)
        g_grad_flux.add_all(k['mpiint/scal_fpts_unpack'])

        # Compute the common solution at our MPI interfaces
        for l in k['mpiint/con_u']:
            g_grad_flux.add(l, deps=deps(l, 'mpiint/scal_fpts_unpack'))

        # Shock capturing kernels which run after the solution exchange
        self.shock_capturing.add_to_graph_post_recv(g_grad_flux, k, deps)

        # Compute the transformed gradient of the partially corrected solution
        g_grad_flux.add_all(k['eles/gradpcoru_upts'])

        # Compute the transformed gradient of the corrected solution
        for l in k['eles/gradcoru_upts']:
            d = deps(l, 'eles/gradpcoru_upts') + k['mpiint/con_u']
            g_grad_flux.add(l, deps=d)

        # Obtain the physical gradients at the solution points
        for l in k['eles/gradcoru_phys_upts']:
            g_grad_flux.add(l, deps=deps(l, 'eles/gradcoru_upts'))

        # Compute the fused transformed flux and corrected gradient
        # (depends on avfill when AV is active — empty list otherwise)
        for l in k['eles/disf_fused']:
            ldeps = deps(l, 'eles/gradcoru_upts') + k['eles/avfill']
            g_grad_flux.add(l, deps=ldeps)

        # Interpolate these gradients to the flux points
        for l in k['eles/gradcoru_fpts']:
            ldeps = deps(l, 'eles/disf_fused', 'eles/gradcoru_phys_upts')
            g_grad_flux.add(l, deps=ldeps)

        # Set dependencies for interface flux interpolation
        ideps = k['eles/gradcoru_fpts'] + k['eles/disf_fused']

        # Pack and send these interpolated gradients to our neighbours
        g_grad_flux.add_all(k['mpiint/vect_fpts_pack'], deps=ideps)
        for send, pack in zip(m['vect_fpts_send'],
                              k['mpiint/vect_fpts_pack']):
            g_grad_flux.add_mpi_req(send, deps=[pack])

        # Compute the common normal flux at our internal/boundary interfaces
        g_grad_flux.add_all(k['iint/comm_flux'],
                            deps=ideps + k['eles/avfill'],
                            pdeps=k['mpiint/vect_fpts_pack'])
        g_grad_flux.add_all(k['bcint/comm_flux'],
                            deps=ideps + k['eles/avfill'])

        # Interpolate the gradients to the quadrature points
        for l in k['eles/gradcoru_qpts']:
            ldeps = deps(l, 'eles/gradcoru_phys_upts')
            g_grad_flux.add(l, deps=ldeps,
                            pdeps=k['mpiint/vect_fpts_pack'])

        # Interpolate the solution to the quadrature points
        g_grad_flux.add_all(k['eles/qptsu'])

        # Compute the transformed flux
        for l in k['eles/disf']:
            if k['eles/qptsu']:
                ldeps = deps(l, 'eles/gradcoru_qpts', 'eles/qptsu')
            elif k['eles/gradcoru_fpts']:
                ldeps = deps(l, 'eles/gradcoru_fpts')
            else:
                ldeps = deps(l, 'eles/gradcoru_phys_upts')
            g_grad_flux.add(l, deps=ldeps + k['eles/avfill'])

        # Compute the transformed divergence of the partially corrected flux
        for l in k['eles/divpcorf']:
            d = deps(l, 'eles/disf', 'eles/disf_fused')
            g_grad_flux.add(l, deps=d)

        kgroup = [
            k['eles/gradpcoru_upts'], k['eles/gradcoru_upts'],
            k['eles/gradcoru_phys_upts'], k['eles/disf_fused'],
            k['eles/gradcoru_fpts'], k['eles/gradcoru_qpts'],
            k['eles/qptsu'], k['eles/disf'], k['eles/divpcorf']
        ]
        for ks in zip_longest(*kgroup):
            # Flux-AA on; inputs to disf and divpcorf are from quad pts
            if k['eles/qptsu']:
                subs = [
                    [(ks[0], 'out'), (ks[1], 'out'), (ks[2], 'gradu'),
                     (ks[4], 'b'), (ks[5], 'b')],
                    [(ks[6], 'out'), (ks[7], 'u')],
                    [(ks[5], 'out'), (ks[7], 'f'), (ks[8], 'b')],
                ]
            # Fusion on; disf_fused replaces disf and gradcoru_phys_upts
            elif k['eles/disf_fused']:
                subs = [
                    [(ks[0], 'out'), (ks[1], 'out'),
                     (ks[3], 'gradu'), (ks[4], 'b')],
                    [(ks[3], 'f'), (ks[8], 'b')],
                ]
            # No flux-AA and no gradient fusion
            else:
                subs = [
                    [(ks[0], 'out'), (ks[1], 'out'), (ks[2], 'gradu'),
                     (ks[4], 'b'), (ks[7], 'f'), (ks[8], 'b')],
                ]

            self._group(g_grad_flux, ks, subs=subs)

        g_grad_flux.commit()

        # Graph: receive MPI gradients, compute MPI flux and divergence
        g_mpi_flux = self.backend.graph()

        # Compute the common normal flux at our MPI interfaces
        # (vect_fpts_unpack may be absent for some interfaces due to
        # the LDG beta parameter, so we cannot zip these lists)
        g_mpi_flux.add_all(k['mpiint/vect_fpts_unpack'])
        for l in k['mpiint/comm_flux']:
            g_mpi_flux.add(l, deps=deps(l, 'mpiint/vect_fpts_unpack'))

        # Compute the transformed divergence of the corrected flux
        g_mpi_flux.add_all(k['eles/divconf'], deps=k['mpiint/comm_flux'])

        # Obtain the physical divergence of the corrected flux
        for l in k['eles/negdivconf']:
            g_mpi_flux.add(l, deps=deps(l, 'eles/divconf'))

        # Group divconf and negdivconf kernels
        for k1, k2 in zip_longest(k['eles/divconf'],
                                   k['eles/negdivconf']):
            self._group(g_mpi_flux, [k1, k2])

        g_mpi_flux.commit()

        return g_soln, g_grad_flux, g_mpi_flux

    @memoize
    def _compute_grads_graph(self, uinbank):
        m = self._mpireqs
        k, *_ = self._get_kernels(uinbank, None)

        def deps(dk, *names): return self._kdeps(k, dk, *names)

        # Graph: interpolate solution, exchange, partial gradient
        g_soln = self.backend.graph()
        g_soln.add_mpi_reqs(m['scal_fpts_recv'])

        # Interpolate the solution to the flux points
        g_soln.add_all(k['eles/disu'])

        # Pack and send these interpolated solutions to our neighbours
        g_soln.add_all(k['mpiint/scal_fpts_pack'], deps=k['eles/disu'])
        for send, pack in zip(m['scal_fpts_send'],
                              k['mpiint/scal_fpts_pack']):
            g_soln.add_mpi_req(send, deps=[pack])

        # Compute the common solution at our internal/boundary interfaces
        g_soln.add_all(k['iint/con_u'], deps=k['eles/disu'])
        g_soln.add_all(k['bcint/con_u'], deps=k['eles/disu'])

        # Compute the transformed gradient of the partially corrected solution
        g_soln.add_all(k['eles/gradpcoru_upts'],
                       deps=k['iint/con_u'] + k['bcint/con_u'])

        g_soln.commit()

        # Graph: receive MPI solution, complete gradient computation
        g_grad = self.backend.graph()

        # Unpack MPI face data (may be empty when unpack is a no-op)
        g_grad.add_all(k['mpiint/scal_fpts_unpack'])

        # Compute the common solution at our MPI interfaces
        for l in k['mpiint/con_u']:
            g_grad.add(l, deps=deps(l, 'mpiint/scal_fpts_unpack'))

        # Compute the transformed gradient of the corrected solution
        g_grad.add_all(k['eles/gradcoru_upts'], deps=k['mpiint/con_u'])

        # Obtain the physical gradients at the solution points
        for l in k['eles/gradcoru_phys']:
            g_grad.add(l, deps=deps(l, 'eles/gradcoru_upts'))

        g_grad.commit()

        return g_soln, g_grad
