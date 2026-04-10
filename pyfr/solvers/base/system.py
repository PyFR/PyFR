from collections import defaultdict
import inspect
import itertools as it
import statistics

import numpy as np

from pyfr.backends.base import NullKernel
from pyfr.cache import memoize
from pyfr.mpiutil import autofree, get_comm_rank_root, mpi
from pyfr.shapes import BaseShape
from pyfr.util import subclasses


class BaseSystem:
    elementscls = None
    intinterscls = None
    mpiinterscls = None
    bbcinterscls = None

    # Nonce sequence
    _nonce_seq = it.count()

    # Extra kernel/MPI providers (overridden by subclasses, e.g., for AV)
    _extra_kern_parts = {}
    _extra_mpi_parts = []

    def __init__(self, backend, mesh, initsoln, registers, cfg, serialiser):
        self.backend = backend
        self.mesh = mesh
        self.cfg = cfg

        # Plugin kernel-creation callbacks
        self._kernel_callbacks = []

        # Conservative and physical variable names
        convars = self.elementscls.convars(mesh.ndims, cfg)
        privars = self.elementscls.privars(mesh.ndims, cfg)

        # Validate the constants block
        for c in cfg.items('constants'):
            if c in convars or c in privars:
                raise ValueError(f'Invalid variable {c!r} in [constants]')

        # Save the number of dimensions and field variables
        self.ndims = mesh.ndims
        self.nvars = len(convars)

        # Obtain a nonce to uniquely identify this system
        self.nonce = nonce = str(next(self._nonce_seq))

        # Load the elements
        eles, elemap, ics = self._load_eles(mesh, initsoln, nonce)
        backend.commit()

        # Retain the element map; this may be deleted by clients
        self.ele_map = elemap

        # Get the types, num DOFs and shapes of the elements
        self.ele_types = list(elemap)
        self.ele_ndofs = [e.neles*e.nupts*e.nvars for e in eles]
        self.ele_shapes = {etype: (e.nupts, e.nvars, e.neles)
                           for etype, e in elemap.items()}

        # Get all the solution point locations for the elements
        self.ele_ploc_upts = [e.ploc_at_np('upts') for e in eles]

        self.eles_vect_upts = None
        if hasattr(eles[0], '_grad_upts'):
            self.eles_vect_upts = [e._grad_upts for e in eles]

        # Allocate register banks (RHS first, then non-RHS)
        self._alloc_register_banks(registers, eles, ics)

        # Load the interfaces
        self._int_inters = self._load_int_inters(mesh, elemap)
        self._mpi_inters = self._load_mpi_inters(mesh, elemap)
        self._bc_inters, self._bc_prefns = self._load_bc_inters(mesh, elemap,
                                                                initsoln,
                                                                serialiser)

    def _alloc_register_banks(self, registers, eles, ics):
        self.ele_banks = [[] for _ in eles]
        self.nrhs = 0

        # Allocate RHS banks first (with initial conditions), then non-RHS
        for rhs in [True, False]:
            for r in registers:
                if r.rhs != rhs or r.dynamic or not r.n:
                    continue

                if rhs:
                    self.nrhs += r.n

                for eidx, (ele, ic) in enumerate(zip(eles, ics)):
                    extent = r.extent or f'bank_{id(ele)}'

                    for _ in range(r.n):
                        m = ele.alloc_bank(extent, ic=ic if rhs else None)
                        self.ele_banks[eidx].append(m)

        self.nother = sum(
            r.n for r in registers
            if not r.rhs and not r.dynamic and r.n
        )

    def register_kernel_callback(self, names, callback):
        # Check for extern name clashes with other plugins
        for cb_names, _ in self._kernel_callbacks:
            if clash := [n for n in names if n in cb_names]:
                raise ValueError(f'Extern name clash: {clash}')

        self._kernel_callbacks.append((tuple(names), callback))

    def _field_view(self, interside, field, layout, view_fn,
                    perm=Ellipsis, vshape=()):
        matmap, rowmap, colmap, reorder = [], [], [], []

        for etype, fidx, eidxs, idx in interside.foreach():
            mat, eles = field[etype], self.ele_map[etype]
            n = len(eidxs)

            if layout == 'fpts':
                fpts = eles.srtd_face_fpts[fidx][eidxs]
                nfp = fpts.shape[1]
                matmap.append(np.full(n * nfp, mat.mid))
                rowmap.append(fpts.ravel())
                colmap.append(np.repeat(eidxs, nfp))
            elif layout == 'face':
                nfp = 1
                matmap.append(np.full(n, mat.mid))
                rowmap.append(np.full(n, fidx))
                colmap.append(eidxs)
            elif layout == 'face-expand':
                nfp = eles.nfacefpts[fidx]
                matmap.append(np.full(n*nfp, mat.mid))
                rowmap.append(np.full(n*nfp, fidx))
                colmap.append(np.repeat(eidxs, nfp))

            reorder.append(np.repeat(idx, nfp))

        ro = np.argsort(np.concatenate(reorder), kind='stable')[perm]
        m = np.concatenate(matmap)[ro]
        r = np.concatenate(rowmap)[ro]
        c = np.concatenate(colmap)[ro]
        return view_fn(m, r, c, vshape=vshape)

    def _compute_perm(self, interside, field):
        # Compute the optimal memory access permutation for a field
        v = self._field_view(interside, field, 'fpts',
                             self.backend.view, vshape=())
        return np.argsort(v.mapping.get()[0])

    def make_field_views(self, field, layout='fpts', bc_layout=None,
                         vshape=()):
        bc_layout = bc_layout or layout
        be = self.backend
        use_perm = lambda l: l in ('fpts', 'face-expand')

        iint_views = []
        for i in self._int_inters:
            perm = i._perm if use_perm(layout) else Ellipsis
            lhs = self._field_view(i.lhs, field, layout, be.view, perm,
                                   vshape)
            rhs = self._field_view(i.rhs, field, layout, be.view, perm,
                                   vshape)
            iint_views.append((lhs, rhs))

        mpi_views = []
        for m in self._mpi_inters:
            lhs = self._field_view(m.lhs, field, layout, be.xchg_view,
                                   vshape=vshape)
            rhs = be.xchg_matrix_for_view(lhs)
            mpi_views.append((lhs, rhs))

        bc_views = []
        for b in self._bc_inters:
            perm = b._perm if use_perm(bc_layout) else Ellipsis
            lhs = self._field_view(b.lhs, field, bc_layout, be.view,
                                   perm, vshape)
            bc_views.append(lhs)

        return iint_views, mpi_views, bc_views

    def register_mpi_exchange(self, name, mpi_views, send=None, recv=None):
        be = self.backend
        comm, rank, root = get_comm_rank_root()

        def register(m, lhs, rhs, tag):
            if not send or send(m):
                m.kernels[f'{name}_pack'] = lambda: be.kernel('pack', lhs)
                m.mpireqs[f'{name}_send'] = lambda: lhs.sendreq(
                    comm, m.rhsrank, tag
                )
            if not recv or recv(m):
                m.kernels[f'{name}_unpack'] = lambda: be.kernel('unpack', rhs)
                m.mpireqs[f'{name}_recv'] = lambda: rhs.recvreq(
                    comm, m.rhsrank, tag
                )

        for m, (lhs, rhs) in zip(self._mpi_inters, mpi_views):
            register(m, lhs, rhs, m.next_mpi_tag())

    def commit(self):
        # Prepare the kernels and any associated MPI requests
        self._gen_kernels(self.nrhs, self.ele_map.values(), self._int_inters,
                          self._mpi_inters, self._bc_inters)
        self._gen_mpireqs(self._mpi_inters)
        self.backend.commit()

        self.has_src_macros = any(eles.has_src_macros
                                  for eles in self.ele_map.values())

        # Delete the memory-intensive ele_map and interface objects
        del self.ele_map
        del self._int_inters
        del self._mpi_inters

        for b in self._bc_inters:
            del b.elemap

        # Observed input/output bank numbers
        self._rhs_uin_fout = set()

    def _load_eles(self, mesh, initsoln, nonce):
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        # Load the elements
        elemap = {etype: self.elementscls(basismap[etype], spts, self.cfg)
                  for etype, spts in mesh.spts.items()}

        eles = list(elemap.values())

        # Compute the initial conditions
        if initsoln:
            ics = [ele.set_ics_from_soln(initsoln.data[et], initsoln.config)
                   for et, ele in elemap.items()]
        else:
            ics = [ele.set_ics_from_cfg() for ele in eles]

        # Allocate these elements on the backend
        for etype, ele in elemap.items():
            curved = mesh.spts_curved[etype]
            linoff = np.max(*np.nonzero(curved), initial=-1) + 1

            ele.set_backend(self.backend, nonce, linoff)

        return eles, elemap, ics

    def _load_int_inters(self, mesh, elemap):
        int_inters = self.intinterscls(self.backend, *mesh.con, elemap,
                                       self.cfg)

        return [int_inters]

    def _load_mpi_inters(self, mesh, elemap):
        mpi_inters = []
        for p, con in mesh.con_p.items():
            mpiiface = self.mpiinterscls(self.backend, con, p, elemap,
                                         self.cfg)
            mpi_inters.append(mpiiface)

        return mpi_inters

    def _load_bc_inters(self, mesh, elemap, initsoln, serialiser):
        comm, rank, root = get_comm_rank_root()

        bccls = self.bbcinterscls
        bcmap = {b.type: b for b in subclasses(bccls, just_leaf=True)}
        bc_inters, bc_prefns = [], {}

        prevcfg = initsoln.config if initsoln else None

        # Iterate over all boundaries in the mesh
        for c in mesh.codec:
            if not c.startswith('bc/'):
                continue

            # Construct an MPI communicator for this boundary
            bname = c.removeprefix('bc/')
            localbc = bname in mesh.bcon
            bccomm = autofree(comm.Split(1 if localbc else mpi.UNDEFINED))

            # Get the class
            cfgsect = f'soln-bcs-{bname}'
            bcclass = bcmap[self.cfg.get(cfgsect, 'type')]

            # Check if there is serialised data for this boundary in initsoln
            sdata = initsoln.state.get(f'bcs/{bname}') if initsoln else None

            # If we have this boundary then create an instance
            if localbc:
                bciface = bcclass(self.backend, mesh.bcon[bname], elemap,
                                  cfgsect, self.cfg, bccomm)
                bciface.setup(sdata, prevcfg)
                bc_inters.append(bciface)
            else:
                bciface = None

            # Allow the boundary to return a preparation callback
            if (pfn := bcclass.preparefn(bciface, mesh, elemap)):
                bc_prefns[bname] = pfn

            bcclass.serialisefn(bciface, f'bcs/{bname}', serialiser)

        return bc_inters, bc_prefns

    def _gen_kernels(self, nregs, eles, iint, mpiint, bcint):
        self._kernels = kernels = defaultdict(list)

        # Helper function to tag the element type/MPI interface
        # associated with a kernel; used for dependency analysis
        self._ktags = {}

        def tag_kern(pname, prov, kern):
            self._ktags[kern] = f'{pname}/{prov.name}'

        provnames = ['eles', 'iint', 'mpiint', 'bcint']
        provlists = [eles, iint, mpiint, bcint]

        for pn, provs in self._extra_kern_parts.items():
            provnames.append(pn)
            provlists.append(provs)

        for pn, provs in zip(provnames, provlists):
            for p in provs:
                for kn, kgetter in p.kernels.items():
                    # Skip private kernels
                    if kn.startswith('_'):
                        continue

                    # See if the kernel depends on uin/fout
                    kparams = inspect.signature(kgetter).parameters
                    if 'uin' in kparams or 'fout' in kparams:
                        for i in range(nregs):
                            kern = kgetter(i)
                            if isinstance(kern, NullKernel):
                                continue

                            if 'uin' in kparams:
                                kernels[f'{pn}/{kn}', i, None].append(kern)
                            else:
                                kernels[f'{pn}/{kn}', None, i].append(kern)

                            tag_kern(pn, p, kern)
                    else:
                        kerns = kgetter()
                        if not isinstance(kerns, list):
                            kerns = [kerns]

                        for kern in kerns:
                            if isinstance(kern, NullKernel):
                                continue

                            kernels[f'{pn}/{kn}', None, None].append(kern)

                            tag_kern(pn, p, kern)

        bindable = [k for ks in kernels.values() for k in ks if k.rtnames]
        for cb_names, cb in self._kernel_callbacks:
            for k in bindable:
                if any(name in cb_names for name in k.rtnames):
                    cb(k)

    def _gen_mpireqs(self, mpiint):
        self._mpireqs = mpireqs = defaultdict(list)

        for m in [*mpiint, *self._extra_mpi_parts]:
            for mn, mgetter in m.mpireqs.items():
                mpireqs[mn].append(mgetter())

    @memoize
    def _get_kernels(self, uinbank, foutbank):
        kernels = defaultdict(list)

        # Filter down the kernels dictionary
        for (kn, ui, fo), k in self._kernels.items():
            if ((ui is None and fo is None) or
                (ui is not None and ui == uinbank) or
                (fo is not None and fo == foutbank)):
                kernels[kn].extend(k)

        # Handle kernels which have arguments that can be bound at runtime
        binders, bckerns = [], defaultdict(dict)
        for kn, kerns in kernels.items():
            for k in kerns:
                if k.rtnames:
                    binders.append(k.bind)

                if kn.startswith('bcint/'):
                    bcname = self._ktags[k].removeprefix('bcint/')
                    bkname = kn.removeprefix('bcint/')

                    bckerns[bcname][bkname] = k

        return kernels, binders, bckerns

    def _kdeps(self, kdict, kern, *dnames):
        deps = []

        for name in dnames:
            for k in kdict[name]:
                if self._ktags[kern] == self._ktags[k]:
                    deps.append(k)

        return deps

    def _prepare_kernels(self, t, uinbank, foutbank):
        _, binders, bckerns = self._get_kernels(uinbank, foutbank)

        for b, bfn in self._bc_prefns.items():
            bfn(self, uinbank, t, bckerns[b])

        for b in binders:
            b(t=t)

    def _rhs_graphs(self, uinbank, foutbank):
        pass

    def rhs(self, t, uinbank, foutbank):
        if uinbank >= self.nrhs or foutbank >= self.nrhs:
            raise ValueError('Invalid register numbers')

        self._rhs_uin_fout.add((uinbank, foutbank))
        self._prepare_kernels(t, uinbank, foutbank)

        for graph in self._rhs_graphs(uinbank, foutbank):
            self.backend.run_graph(graph)

    def _preproc_graphs(self, uinbank):
        return ()

    def preproc(self, t, uinbank):
        if uinbank >= self.nrhs:
            raise ValueError('Invalid register number')

        self._prepare_kernels(t, uinbank, None)

        for graph in self._preproc_graphs(uinbank):
            self.backend.run_graph(graph)

    def postproc(self, uinbank):
        pass

    def rhs_wait_times(self):
        # Group together timings for graphs which are semantically equivalent
        times = defaultdict(list)
        for u, f in self._rhs_uin_fout:
            for i, g in enumerate(self._rhs_graphs(u, f)):
                times[i].extend(g.get_wait_times())

        # Compute the mean and standard deviation
        stats = []
        for t in times.values():
            mean = statistics.mean(t) if t else 0
            stdev = statistics.stdev(t, mean) if len(t) >= 2 else 0
            median = statistics.median(t) if t else 0

            stats.append((mean, stdev, median))

        return stats

    def _compute_grads_graph(self, uinbank):
        raise NotImplementedError(f'Solver {self.name!r} does not compute '
                                  'corrected gradients of the solution')

    def compute_grads(self, t, uinbank):
        if uinbank >= self.nrhs:
            raise ValueError('Invalid register number')

        self._prepare_kernels(t, uinbank, None)

        for graph in self._compute_grads_graph(uinbank):
            self.backend.run_graph(graph)

    def evalsrcmacros(self, uinoutbank):
        kkey = ('eles/evalsrcmacros', uinoutbank, None)

        self.backend.run_kernels(self._kernels[kkey])

    def ele_scal_upts(self, idx):
        return [eb[idx].get() for eb in self.ele_banks]

    def _group(self, g, kerns, subs=[]):
        # Eliminate non-existent kernels
        kerns = [k for k in kerns if k is not None]

        # Eliminate substitutions associated with non-existent kernels
        subs = [[(k, n) for k, n in sub if k] for sub in subs]
        subs = [sub for sub in subs if len(sub) > 1]

        g.group(kerns, subs)
