import numpy as np

from pyfr.solvers.base.elements import ExportableField


class EntropyFilter:
    def __init__(self, backend, cfg, system, int_inters, mpi_inters,
                 bc_inters):
        self._be = backend

        # Register pointwise kernel templates
        kprefix = f'pyfr.solvers.{system.ef_solver}.kernels'
        backend.pointwise.register(f'{kprefix}.entropylocal')
        backend.pointwise.register(f'{kprefix}.entropyfilter')

        # Per-element-type setup
        self._entmin = {}
        for etype, eles in system.ele_map.items():
            if eles.basis.order > 0:
                self._setup_etype(etype, eles, cfg, system.nonce)

        backend.commit()

        # Create interface views and register comm_entropy kernels
        self._setup_interfaces(system, int_inters, mpi_inters, bc_inters)

    def _setup_etype(self, etype, eles, cfg, nonce):
        be = self._be
        nfaces = len(eles.nfacefpts)
        neles = eles.neles

        # Allocate one minimum entropy value per face
        entmin_int = np.full((nfaces, neles), -be.fpdtype_max,
                             dtype=be.fpdtype)
        entmin = be.matrix((nfaces, neles), tags={'align'},
                            extent=nonce + 'entmin_int',
                            initval=entmin_int)
        self._entmin[etype] = entmin

        # Allocate space for filter strength (1 = no filter, 0 = max)
        ef_filter = be.matrix((1, neles),
                               extent=nonce + 'ef_filter',
                               tags={'align'})

        # Register exportable field for filter strength
        eles.export_fields.append(ExportableField(
            name='ef-filter', shape=(),
            getter=lambda: ef_filter.get()[0]
        ))

        # Setup nodal/modal operator matrices
        form = cfg.get('solver-entropy-filter', 'formulation', 'nonlinear')
        invvdm, vdm_ef = self._build_operators(eles, form)

        if eles.basis.fpts_in_upts:
            m0 = None
        else:
            m0 = be.const_matrix(eles.basis.m0)

        # Build template arguments
        eftplargs = self._build_tplargs(eles, cfg, nfaces, form)

        # Register kernel factories on elements
        def local_entropy_kern(uin):
            return be.kernel(
                'entropylocal', tplargs=eftplargs, dims=[eles.neles],
                u=eles.scal_upts[uin], entmin_int=entmin, m0=m0
            )

        def entropy_filter_kern(uin):
            return be.kernel(
                'entropyfilter', tplargs=eftplargs, dims=[eles.neles],
                u=eles.scal_upts[uin], entmin_int=entmin, ef_filter=ef_filter,
                vdm=vdm_ef, invvdm=invvdm, m0=m0,
                mean_wts=eles.mean_wts
            )

        eles.kernels['local_entropy'] = local_entropy_kern
        eles.kernels['entropy_filter'] = entropy_filter_kern

    def _build_operators(self, eles, form):
        be = self._be

        if form == 'linearised':
            return None, None
        elif form == 'nonlinear':
            invvdm = be.const_matrix(eles.basis.ubasis.invvdm.T)
            vdm = eles.basis.ubasis.vdm.T

            if not eles.basis.fpts_in_upts:
                vdmf = eles.basis.ubasis.vdm_at(eles.basis.fpts).T
                vdm = np.vstack([vdm, vdmf])

            return invvdm, be.const_matrix(vdm)
        else:
            raise ValueError('Invalid entropy filter formulation.')

    def _build_tplargs(self, eles, cfg, nfaces, form):
        fpts_in_upts = eles.basis.fpts_in_upts
        nefpts = eles.nupts if fpts_in_upts else eles.nupts + eles.nfpts
        ub = eles.basis.ubasis

        return {
            'ndims': eles.ndims, 'nupts': eles.nupts,
            'nfpts': eles.nfpts, 'nefpts': nefpts,
            'nvars': eles.nvars, 'nfaces': nfaces,
            'c': cfg.items_as('constants', float),
            'order': eles.basis.order,
            'fpts_in_upts': fpts_in_upts,
            'd_min': cfg.getfloat('solver-entropy-filter', 'd-min', 1e-6),
            'p_min': cfg.getfloat('solver-entropy-filter', 'p-min', 1e-6),
            'e_tol': cfg.getfloat('solver-entropy-filter', 'e-tol', 1e-6),
            'f_tol': cfg.getfloat('solver-entropy-filter', 'f-tol', 1e-4),
            'niters': cfg.getfloat('solver-entropy-filter', 'niters', 2),
            'linearise': form == 'linearised',
            'ubdegs': [int(max(dd)) for dd in ub.degrees],
        }

    def _setup_interfaces(self, system, int_inters, mpi_inters, bc_inters):
        be = self._be

        # Create interface views for entmin using system API
        iint_v, mpi_v, bc_v = system.make_field_views(
            self._entmin, layout='face', bc_layout='face-expand'
        )

        # Register MPI exchange for entropy face values
        system.register_mpi_exchange('ent_fpts', mpi_v)

        # Register comm_entropy kernels on internal/MPI interfaces
        def cent_kern(kname, intf, lhs, rhs):
            return lambda: be.kernel(kname, tplargs={}, dims=[intf.ninters],
                                     entmin_lhs=lhs, entmin_rhs=rhs)

        kprefix = 'pyfr.solvers.baseadvec.kernels'
        be.pointwise.register(f'{kprefix}.intcent')
        for i, (lhs, rhs) in zip(int_inters, iint_v):
            i.kernels['comm_entropy'] = cent_kern('intcent', i, lhs, rhs)

        be.pointwise.register(f'{kprefix}.mpicent')
        for m, (lhs, rhs) in zip(mpi_inters, mpi_v):
            m.kernels['comm_entropy'] = cent_kern('mpicent', m, lhs, rhs)

        # Register comm_entropy kernels on BC interfaces
        for b, lhs in zip(bc_inters, bc_v):
            b.kernels['comm_entropy'] = b.comm_entropy_kernel(lhs)

    def add_to_graph_pre_recv(self, g, k, m):
        # Post entropy MPI receives
        g.add_mpi_reqs(m['ent_fpts_recv'])

        # Run the entropy filter
        g.add_all(k['eles/entropy_filter'])

        # Pack and send entropy face values to neighbours
        g.add_all(k['mpiint/ent_fpts_pack'], deps=k['eles/entropy_filter'])
        for send, pack in zip(m['ent_fpts_send'], k['mpiint/ent_fpts_pack']):
            g.add_mpi_req(send, deps=[pack])

        # Compute common entropy at internal interfaces
        g.add_all(k['iint/comm_entropy'],
                  deps=k['eles/entropy_filter'] + k['mpiint/ent_fpts_pack'])

        # BC entropy needs the solution at flux points (disu)
        g.add_all(k['bcint/comm_entropy'], deps=k['eles/disu'])

    def add_to_graph_post_recv(self, g, k, deps):
        # Unpack MPI entropy data (may be empty when unpack is a no-op)
        g.add_all(k['mpiint/ent_fpts_unpack'])

        # Compute common entropy at MPI interfaces
        for c in k['mpiint/comm_entropy']:
            g.add(c, deps=deps(c, 'mpiint/ent_fpts_unpack'))

    def preproc_graphs(self, be, k, m, deps):
        # Graph: filter, compute local entropy, exchange, internal/BC entropy
        g_filter = be.graph()
        g_filter.add_mpi_reqs(m['ent_fpts_recv'])

        # Run entropy filter then compute local entropy minima
        g_filter.add_all(k['eles/entropy_filter'])
        g_filter.add_all(k['eles/local_entropy'],
                         deps=k['eles/entropy_filter'])

        # Interpolate to flux points (needed by bcint/comm_entropy)
        g_filter.add_all(k['eles/disu'], deps=k['eles/entropy_filter'])

        # Pack and send entropy values to neighbours
        g_filter.add_all(k['mpiint/ent_fpts_pack'],
                         deps=k['eles/local_entropy'])
        for send, pack in zip(m['ent_fpts_send'], k['mpiint/ent_fpts_pack']):
            g_filter.add_mpi_req(send, deps=[pack])

        # Compute common entropy minima at internal/boundary interfaces
        g_filter.add_all(k['iint/comm_entropy'], deps=k['eles/local_entropy'])
        g_filter.add_all(k['bcint/comm_entropy'],
                         deps=k['eles/local_entropy'] + k['eles/disu'])
        g_filter.commit()

        # Graph: MPI comm_entropy (if we have MPI interfaces)
        if k['mpiint/comm_entropy']:
            g_mpi_ent = be.graph()
            g_mpi_ent.add_all(k['mpiint/ent_fpts_unpack'])
            for c in k['mpiint/comm_entropy']:
                g_mpi_ent.add(c, deps=deps(c, 'mpiint/ent_fpts_unpack'))

            g_mpi_ent.commit()
            return g_filter, g_mpi_ent

        return g_filter,

    def postproc(self, be, k):
        be.run_kernels(k['eles/entropy_filter'])
