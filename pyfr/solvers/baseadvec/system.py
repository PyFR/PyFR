from itertools import zip_longest

from pyfr.cache import memoize
from pyfr.solvers.base import BaseSystem
from pyfr.solvers.baseadvec.entfilter import EntropyFilter


class BaseAdvectionSystem(BaseSystem):
    _shock_capturing_modes = {'none', 'entropy-filter'}

    def __init__(self, backend, mesh, initsoln, nregs, cfg, serialiser,
                 *, needs_cfl=False):
        super().__init__(backend, mesh, initsoln, nregs, cfg, serialiser)
        self._needs_cfl = needs_cfl

        if needs_cfl:
            for eles in self.ele_map.values():
                eles.init_wavespeed()

    def commit(self):
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')

        # Create scal_fpts views and MPI exchange at the system level
        scal_fpts = {et: e._scal_fpts for et, e in self.ele_map.items()}
        iint_v, mpi_v, bc_v = self.make_field_views(
            scal_fpts, vshape=(self.nvars,)
        )
        for i, (lhs, rhs) in zip(self._int_inters, iint_v):
            i.scal_lhs = lhs
            i.scal_rhs = rhs
        for m, (lhs, rhs) in zip(self._mpi_inters, mpi_v):
            m.scal_lhs = lhs
            m.scal_rhs = rhs
        for b, lhs in zip(self._bc_inters, bc_v):
            b.scal_lhs = lhs
        self.register_mpi_exchange('scal_fpts', mpi_v)

        if shock_capturing == 'entropy-filter':
            self._ef = EntropyFilter(
                self.backend, self.cfg, self,
                self._int_inters, self._mpi_inters, self._bc_inters
            )
        else:
            self._ef = None

        self.backend.commit()

        # Reduction kernels to find max wavespeed across each element type
        if self._needs_cfl:
            self._wspd_red_kerns = [
                self.backend.kernel('reduction', 'max', ['x'], {'x': e._wspd})
                for e in self.ele_map.values()
            ]

        super().commit()

    @memoize
    def _rhs_graphs(self, uinbank, foutbank):
        m = self._mpireqs
        k, *_ = self._get_kernels(uinbank, foutbank)

        def deps(dk, *names): return self._kdeps(k, dk, *names)

        # Graph 1: interpolate solution, exchange, compute local flux
        g_intf = self.backend.graph()
        g_intf.add_mpi_reqs(m['scal_fpts_recv'])

        # Interpolate the solution to the flux points
        g_intf.add_all(k['eles/disu'], deps=k['eles/entropy_filter'])

        # EF adds its kernels (topo sort handles ordering)
        if self._ef:
            self._ef.add_to_graph_pre_recv(g_intf, k, m)

        # Pack and send these interpolated solutions to our neighbours
        g_intf.add_all(k['mpiint/scal_fpts_pack'], deps=k['eles/disu'])
        for send, pack in zip(m['scal_fpts_send'],
                              k['mpiint/scal_fpts_pack']):
            g_intf.add_mpi_req(send, deps=[pack])

        # Compute the common normal flux at our internal/boundary interfaces
        g_intf.add_all(k['iint/comm_flux'],
                       deps=k['eles/disu'] + k['mpiint/scal_fpts_pack'])
        g_intf.add_all(k['bcint/comm_flux'],
                       deps=k['eles/disu'] + k['bcint/comm_entropy'])

        # Make a copy of the solution (if used by source terms)
        g_intf.add_all(k['eles/copy_soln'], deps=k['eles/entropy_filter'])

        g_intf.commit()

        # Graph 2: receive MPI solution, compute flux and divergence
        g_flux_div = self.backend.graph()

        # Interpolate the solution to the quadrature points
        g_flux_div.add_all(k['eles/qptsu'])

        # Compute the transformed flux
        for l in k['eles/tdisf']:
            g_flux_div.add(l, deps=deps(l, 'eles/qptsu'))

        # Compute the transformed divergence of the partially corrected flux
        for l in k['eles/tdivtpcorf']:
            g_flux_div.add(l, deps=deps(l, 'eles/tdisf'))

        # Unpack MPI face data (may be empty when unpack is a no-op)
        g_flux_div.add_all(k['mpiint/scal_fpts_unpack'])

        # Compute the common normal flux at our MPI interfaces
        for l in k['mpiint/comm_flux']:
            g_flux_div.add(l, deps=deps(l, 'mpiint/scal_fpts_unpack'))

        # EF: unpack and compute comm_entropy at MPI interfaces
        if self._ef:
            self._ef.add_to_graph_post_recv(g_flux_div, k, deps)

        # Compute the transformed divergence of the corrected flux
        for l in k['eles/tdivtconf']:
            ldeps = deps(l, 'eles/tdivtpcorf') + k['mpiint/comm_flux']
            g_flux_div.add(l, deps=ldeps)

        # Obtain the physical divergence of the corrected flux
        for l in k['eles/negdivconf']:
            g_flux_div.add(l, deps=deps(l, 'eles/tdivtconf'))

        kgroup = [k['eles/qptsu'], k['eles/tdisf'], k['eles/tdivtpcorf'],
                  k['eles/tdivtconf'], k['eles/negdivconf']]
        for ks in zip_longest(*kgroup):
            self._group(g_flux_div, ks, subs=[
                [(ks[0], 'out'), (ks[1], 'u')],
                [(ks[1], 'f'), (ks[2], 'b')],
            ])

        g_flux_div.commit()

        return g_intf, g_flux_div

    def _preproc_graphs(self, uinbank):
        if self._ef:
            return self._preproc_graphs_ef(uinbank)
        return ()

    @memoize
    def _preproc_graphs_ef(self, uinbank):
        m = self._mpireqs
        k, *_ = self._get_kernels(uinbank, None)
        def deps(dk, *names): return self._kdeps(k, dk, *names)
        return self._ef.preproc_graphs(self.backend, k, m, deps)

    def postproc(self, uinbank):
        if self._ef:
            if uinbank >= self.nrhs:
                raise ValueError('Invalid register number')
            k, *_ = self._get_kernels(uinbank, None)
            self._ef.postproc(self.backend, k)

    def compute_max_wavespeed(self, uinbank):
        k, *_ = self._get_kernels(uinbank, None)
        kerns = k['eles/wavespeed'] + self._wspd_red_kerns
        self.backend.run_kernels(kerns, wait=True)
        return max(k.retval[0] for k in self._wspd_red_kerns)
