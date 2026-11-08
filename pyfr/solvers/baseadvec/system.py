from pyfr.cache import memoize
from pyfr.solvers.base import BaseSystem
from pyfr.solvers.baseadvec.shockcapturing import get_shock_capturing


class BaseAdvectionSystem(BaseSystem):
    def __init__(self, backend, mesh, initsoln, registers, cfg, serialiser,
                 *, needs_cfl=False):
        super().__init__(backend, mesh, initsoln, registers, cfg, serialiser)
        self._needs_cfl = needs_cfl

        if needs_cfl:
            for eles in self.ele_map.values():
                eles.init_wavespeed()


        # Create scal_fpts views and MPI exchange at the system level
        scal_fpts = {et: e._scal_fpts for et, e in self.ele_map.items()}
        iint_v, mpi_v, bc_v = self.make_field_views(scal_fpts,
                                                    vshape=(self.nvars,))
        for i, lhs, rhs in iint_v:
            i.scal_lhs = lhs
            i.scal_rhs = rhs
        for m, lhs, rhs in mpi_v:
            m.scal_lhs = lhs
            m.scal_rhs = rhs
        for b, lhs in bc_v:
            b.scal_lhs = lhs
        self.register_mpi_exchange('scal_fpts', mpi_v)

        # Instantiate the shock capturing scheme
        sc = get_shock_capturing(self.backend, self.cfg, self)
        self.shock_capturing = sc

        # Take on its kernel/MPI providers
        self._extra_kern_parts = sc.kern_parts
        self._extra_mpi_parts = sc.mpi_parts

    def rewrite_bc_exprs(self, fn=None, externs={}):
        for b in self._bc_inters:
            b.rewrite_exprs(fn, externs)

    def commit(self):
        # Ensure the boundary expressions only reference known variables
        for b in self._bc_inters:
            b.check_exprs()

        self.backend.commit()

        # Reduction kernels to find max wavespeed across each element type
        if self._needs_cfl:
            self._wspd_red_kerns = [
                self.backend.kernel('reduction', 'max', ['x'], {'x': e._wspd})
                for e in self.ele_map.values()
            ]

        super().commit()

        # Populate the per-element wavespeed
        if self._needs_cfl:
            self.compute_max_wavespeed(0)

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

        # Shock capturing kernels which run before the solution exchange
        self.shock_capturing.add_to_graph_pre_recv(g_intf, k, m)

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

        g_intf.commit()

        # Graph 2: receive MPI solution, compute flux and divergence
        g_flux_div = self.backend.graph()

        # Interpolate the solution to the quadrature points
        g_flux_div.add_all(k['eles/qptsu'])

        # Compute the transformed flux
        for l in k['eles/disf']:
            g_flux_div.add(l, deps=deps(l, 'eles/qptsu'))

        # Compute the transformed divergence of the partially corrected flux
        for l in k['eles/divpcorf']:
            g_flux_div.add(l, deps=deps(l, 'eles/disf'))

        # Unpack MPI face data (may be empty when unpack is a no-op)
        g_flux_div.add_all(k['mpiint/scal_fpts_unpack'])

        # Compute the common normal flux at our MPI interfaces
        for l in k['mpiint/comm_flux']:
            g_flux_div.add(l, deps=deps(l, 'mpiint/scal_fpts_unpack'))

        # Shock capturing kernels which run after the solution exchange
        self.shock_capturing.add_to_graph_post_recv(g_flux_div, k, deps)

        # Compute the transformed divergence of the corrected flux
        for l in k['eles/divconf']:
            ldeps = deps(l, 'eles/divpcorf') + k['mpiint/comm_flux']
            g_flux_div.add(l, deps=ldeps)

        # Obtain the physical divergence of the corrected flux
        for l in k['eles/negdivconf']:
            g_flux_div.add(l, deps=deps(l, 'eles/divconf'))

        kgroup = ['eles/qptsu', 'eles/disf', 'eles/divpcorf',
                  'eles/divconf', 'eles/negdivconf']
        subs = [
            [('eles/qptsu', 'out'), ('eles/disf', 'u')],
            [('eles/disf', 'f'), ('eles/divpcorf', 'b')]
        ]
        for ks in self._kzip(k, *kgroup):
            self._group(g_flux_div, ks, subs)

        g_flux_div.commit()

        return g_intf, g_flux_div

    @memoize
    def _preproc_graphs(self, uinbank):
        m = self._mpireqs
        k, *_ = self._get_kernels(uinbank, None)
        def deps(dk, *names): return self._kdeps(k, dk, *names)
        return self.shock_capturing.preproc_graphs(self.backend, k, m, deps)

    def postproc(self, uinbank):
        if uinbank >= self.nrhs:
            raise ValueError('Invalid register number')

        k, *_ = self._get_kernels(uinbank, None)
        self.shock_capturing.postproc(self.backend, k)

    def compute_max_wavespeed(self, uinbank):
        k, *_ = self._get_kernels(uinbank, None)
        kerns = k['eles/wavespeed'] + self._wspd_red_kerns
        self.backend.run_kernels(kerns, wait=True)
        return max(k.retval[0] for k in self._wspd_red_kerns)
