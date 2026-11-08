from weakref import WeakKeyDictionary

import numpy as np

from pyfr.bases import get_basis
from pyfr.mpiutil import autofree, get_comm_rank_root
from pyfr.solvers.base.elements import ExportableField
from pyfr.solvers.baseadvec.shockcapturing import BaseShockCapturing


class _VtxPeer:
    def __init__(self, be, vtx_comm, rank, xv, rm, mv, n):
        self.name = f'p{rank}'

        self.kernels = {
            'pack': lambda: be.kernel('pack', xv),
            'unpack': lambda: be.kernel('unpack', rm),
            'merge': lambda: be.kernel(
                'vtxreduce', tplargs={}, dims=[n], vtx=mv, recv=rm
            ),
        }
        self.mpireqs = {
            'vtx_send': lambda: xv.sendreq(vtx_comm, rank, 0),
            'vtx_recv': lambda: rm.recvreq(vtx_comm, rank, 0),
        }


class ArtificialViscosity(BaseShockCapturing):
    name = 'av'
    scheme = 'artificial-viscosity'
    needs_grad_soln = True

    def __init__(self, backend, cfg, system):
        self._be = backend
        self._mesh = mesh = system.mesh

        # Owned extents for the AV buffers (self-committed, not global)
        self._vtx_ext = backend.extent()
        self._fpts_ext = backend.extent()

        # Allocate vertex buffer
        self._vtx_buf = backend.matrix((1, len(mesh.node_idxs)),
                                       extent=self._vtx_ext)

        # Commit to give vtx_buf basedata (needed for view creation)
        backend.commit_extent(self._vtx_ext)

        # Register kernel templates
        kprefix = 'pyfr.solvers.baseadvecdiff.kernels'
        backend.pointwise.register(f'{kprefix}.shocksensor')
        backend.pointwise.register(f'{kprefix}.avfill')

        # AV config constants
        c_av = cfg.items_as('solver-artificial-viscosity', float)

        # Per-element-type setup
        self.artvisc_fpts = {}
        self._vtx_views = {}
        for etype, eles in system.ele_map.items():
            self._setup_etype(etype, eles, mesh, c_av)

        # Commit the flux point buffers so interface views can be built
        backend.commit_extent(self._fpts_ext)

        # Create interface views and register the vertex exchange
        self._setup_interfaces(system)
        self._setup_mpi()

    def _setup_etype(self, etype, eles, mesh, c_av):
        be = self._be
        basis = eles.basis
        nverts = len(basis.linspts)
        enodes = mesh.spts_nodes[etype]

        # Find which shape point indices correspond to vertices
        spts = np.array(basis.spts)
        linspts = np.array(basis.linspts)
        vtx_idxs = np.argmin(
            np.linalg.norm(spts - linspts[:, None], axis=2), axis=1
        )
        vnodes = enodes[:, vtx_idxs]

        # Allocate artvisc at fpts (populated by avfill kernel)
        artvisc_fpts = be.matrix(
            (eles.nfpts, eles.neles), tags={'align'},
            extent=self._fpts_ext
        )
        self.artvisc_fpts[etype] = artvisc_fpts

        # Linear interpolation basis for vertex → point interpolation
        linbasis = basis.lbasis

        # Build vertex views (full view + per-region views)
        vtx_view = self._setup_vtx_views(etype, eles, mesh, nverts, vnodes)

        # Register kernel factories
        self._setup_kernels(eles, vtx_view, c_av, nverts, vnodes, linbasis,
                            artvisc_fpts)

        # Set us as the shock capturing scheme
        eles.set_shock_capturing(self)

    def _setup_vtx_views(self, etype, eles, mesh, nverts, vnodes):
        be = self._be

        # Build mapping arrays for vertex views
        n = eles.neles * nverts
        matmap = np.full(n, self._vtx_buf.mid)
        rmap = np.zeros(n, dtype=int)
        cmap = np.searchsorted(mesh.node_idxs, vnodes.ravel())

        # Full view for 1D kernels (shocksensor, avfill)
        vmap = be.view_mapping(matmap, rmap, cmap)
        vtx_view = be.view(vmap)

        # Per-region views for 2D tflux kernel
        self._vtx_views[etype] = vtx_views = {}
        for rgn in ('curved', 'linear'):
            if rgn in eles.mesh_regions:
                off = 0 if rgn == 'curved' else eles.linoff
                slc = slice(off*nverts, (off + eles.mesh_regions[rgn])*nverts)
                vtx_views[rgn] = be.view(vmap[slc])

        return vtx_view

    def _setup_kernels(self, eles, vtx_view, c_av, nverts, vnodes, linbasis,
                       artvisc_fpts):
        be = self._be

        # Sensor template arguments
        tplargs_sensor = self._sensor_tplargs(eles, c_av)
        tplargs_sensor['nverts'] = nverts

        # Avfill template arguments
        tplargs_avfill = dict(
            nverts=nverts, nfpts=eles.nfpts,
            av_op=linbasis.nodal_basis_at(eles.basis.fpts).tolist()
        )

        # Register kernel factories on elements
        def shocksensor_kern(uin):
            return be.kernel(
                'shocksensor', tplargs=tplargs_sensor, dims=[eles.neles],
                u=eles.scal_upts[uin], vtx=vtx_view
            )

        def avfill_kern():
            return be.kernel(
                'avfill', tplargs=tplargs_avfill, dims=[eles.neles],
                vtx=vtx_view, artvisc_fpts=artvisc_fpts
            )

        eles.kernels['shocksensor'] = shocksensor_kern
        eles.kernels['avfill'] = avfill_kern

        # Register the exportable AV field (P1 vertex values)
        eles.export_fields.append(ExportableField(
            name='artvisc', shape=(nverts,),
            getter=lambda: self._av_at_vtx(vnodes)
        ))

    def _av_at_vtx(self, vnodes):
        vtx = self._vtx_buf.get().ravel()
        nidxs = self._mesh.node_idxs

        # Gather vertex values: (neles, nverts)
        vidxs = np.searchsorted(nidxs, vnodes.ravel())
        return vtx[vidxs].reshape(vnodes.shape)

    def _sensor_tplargs(self, eles, c_av):
        shockvar = eles.convars.index(eles.shockvar)

        ubname = eles.basis.ubasis.name
        ubdegs = eles.basis.ubasis.degrees
        uborder = eles.basis.ubasis.order

        lubdegs = get_basis(ubname, max(0, uborder - 1)).degrees
        ind_modes = [d not in lubdegs for d in ubdegs]

        mode_degs = [max(d) for d in ubdegs]
        ndeg = uborder + 1

        if uborder >= 2:
            fit_degs = np.arange(1.0, ndeg)
            x = np.log(fit_degs)
            s_weights = np.linalg.pinv(np.vander(x, 2))[0].tolist()

            # Perfect modal decay |b_d|^2 ~ d^-2p normalised to unit energy
            bd2 = fit_degs**(-2*uborder)
            baseline_decay = (bd2 / bd2.sum()).tolist()
        else:
            s_weights = None
            baseline_decay = None

        return dict(
            nvars=eles.nvars, nupts=eles.nupts, svar=shockvar,
            c=c_av, order=eles.basis.order, ind_modes=ind_modes,
            invvdm=eles.basis.ubasis.invvdm.T,
            mode_degs=mode_degs, ndeg=ndeg, s_weights=s_weights,
            baseline_decay=baseline_decay
        )

    def _setup_interfaces(self, system):
        # Create artvisc_fpts views on interfaces for comm_flux
        iint_v, mpi_v, bc_v = system.make_field_views(self.artvisc_fpts)

        # C0 AV is continuous; both sides see the same value
        self._inters_views = WeakKeyDictionary()
        for i, lhs, _ in iint_v:
            self._inters_views[i] = lhs
        for m, lhs, _ in mpi_v:
            self._inters_views[m] = lhs
        for b, lhs in bc_v:
            self._inters_views[b] = lhs

        # Set us as the shock capturing scheme
        for inters in self._inters_views:
            inters.set_shock_capturing(self)

    def _setup_mpi(self):
        be = self._be
        mesh = self._mesh

        # Duplicate the communicator to isolate vertex exchange tags
        comm, _, _ = get_comm_rank_root()
        self._vtx_comm = autofree(comm.Dup())

        # Register vtxreduce kernel template
        kprefix = 'pyfr.solvers.baseadvecdiff.kernels'
        be.pointwise.register(f'{kprefix}.vtxreduce')

        # Create per-peer exchange objects
        self._peers = []
        for rank, shared_global in mesh.shared_nodes.by_rank.items():
            shared_local = np.searchsorted(mesh.node_idxs, shared_global)
            n = len(shared_local)

            matmap = np.full(n, self._vtx_buf.mid)
            rmap = np.zeros(n, dtype=int)
            cmap = shared_local

            vmap = be.view_mapping(matmap, rmap, cmap)
            xv = be.xchg_view(vmap)
            rm = be.xchg_matrix_for_view(xv)
            mv = be.view(vmap)

            self._peers.append(
                _VtxPeer(be, self._vtx_comm, rank, xv, rm, mv, n)
            )

        # Global kernels (not per-peer)
        self.kernels = {
            'vtx_zero': lambda: be.kernel('zero', self._vtx_buf),
        }
        self.mpireqs = {}

        # Provider lists for the system
        self.kern_parts = {'vtx': [self] + self._peers}
        self.mpi_parts = list(self._peers)

    # Shock capturing interface
    def flux_args(self, eles, rgn, pts):
        # Vertex values and their interpolation operator to these points
        return {'artvisc_vtx': self._vtx_views[eles.name][rgn],
                'avop': eles.vtx_interp_at(pts)}

    def comm_flux_args(self, inters):
        return {'artvisc': self._inters_views[inters]}

    def add_to_graph_pre_recv(self, g, k, m):
        # Vertex exchange: post receives, zero buffer, run sensor, pack, send
        g.add_mpi_reqs(m['vtx_recv'])
        g.add_all(k['vtx/vtx_zero'])
        g.add_all(k['eles/shocksensor'], deps=k['vtx/vtx_zero'])
        g.add_all(k['vtx/pack'], deps=k['eles/shocksensor'])
        for send, pack in zip(m['vtx_send'], k['vtx/pack']):
            g.add_mpi_req(send, deps=[pack])

    def add_to_graph_post_recv(self, g, k, deps):
        # Unpack received vertex data (may be empty when unpack is a no-op)
        g.add_all(k['vtx/unpack'])

        # Merge per-peer vertex data (fine-grained: each merge waits
        # only on its own peer's unpack)
        for m in k['vtx/merge']:
            g.add(m, deps=deps(m, 'vtx/unpack'))

        # Fill artvisc_fpts once all merges are complete
        g.add_all(k['eles/avfill'], deps=k['vtx/merge'])
