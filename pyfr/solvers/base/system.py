from collections import defaultdict
import inspect
import itertools as it
import statistics

import numpy as np

from pyfr.backends.base import NullKernel
from pyfr.cache import memoize
from pyfr.shapes import BaseShape
from pyfr.util import subclasses


class BaseSystem:
    elementscls = None
    intinterscls = None
    mpiinterscls = None
    bbcinterscls = None

    # Nonce sequence
    _nonce_seq = it.count()

    def __init__(self, backend, mesh, initsoln, nregs, cfg):
        self.backend = backend
        self.mesh = mesh
        self.cfg = cfg
        self.nregs = nregs

        # Conservative and physical variable names
        convars = self.elementscls.convars(mesh.ndims, cfg)
        privars = self.elementscls.privars(mesh.ndims, cfg)

        # Validate the constants block
        for c in cfg.items('constants'):
            if c in convars or c in privars:
                raise ValueError(f'Invalid variable "{c}" in [constants]')

        # Save the number of dimensions and field variables
        self.ndims = mesh.ndims
        self.nvars = len(convars)

        # Obtain a nonce to uniquely identify this system
        nonce = str(next(self._nonce_seq))

        # Load the elements
        eles, elemap = self._load_eles(mesh, initsoln, nregs, nonce)
        backend.commit()

        # Retain the element map; this may be deleted by clients
        self.ele_map = elemap

        # Get the banks, types, num DOFs and shapes of the elements
        self.ele_banks = [e.scal_upts for e in eles]
        self.ele_types = list(elemap)
        self.ele_ndofs = [e.neles*e.nupts*e.nvars for e in eles]
        self.ele_shapes = {etype: (e.nupts, e.nvars, e.neles)
                           for etype, e in elemap.items()}

        # Get all the solution point locations for the elements
        self.ele_ploc_upts = [e.ploc_at_np('upts') for e in eles]

        if hasattr(eles[0], '_grad_upts'):
            self.eles_vect_upts = [e._grad_upts for e in eles]

        if hasattr(eles[0], 'entmin_int'):
            self.eles_entmin_int = [e.entmin_int for e in eles]

        # Load the interfaces
        self._int_inters = self._load_int_inters(mesh, elemap)
        self._mpi_inters = self._load_mpi_inters(mesh, elemap)
        self._bc_inters = self._load_bc_inters(mesh, elemap)
        backend.commit()

    def commit(self):
        # Prepare the kernels and any associated MPI requests
        self._gen_kernels(self.nregs, self.ele_map.values(), self._int_inters,
                          self._mpi_inters, self._bc_inters)
        self._gen_mpireqs(self._mpi_inters)
        self.backend.commit()

        self.has_src_macros = any(eles.has_src_macros
                                  for eles in self.ele_map.values())

        # Delete the memory-intensive ele_map
        del self.ele_map

        # Save the BC interfaces, but delete the memory-intensive elemap
        for b in self._bc_inters:
            del b.elemap

        # Observed input/output bank numbers
        self._rhs_uin_fout = set()

    def _load_eles(self, mesh, initsoln, nregs, nonce):
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        # Load the elements
        elemap = {etype: self.elementscls(basismap[etype], spts, self.cfg)
                  for etype, spts in mesh.spts.items()}

        eles = list(elemap.values())

        # Set the initial conditions
        if initsoln:
            # Load the config and stats files from the solution
            solncfg = initsoln['config']
            solnsts = initsoln['stats']

            # Get the names of the conserved variables (fields)
            solnfields = solnsts.get('data', 'fields').split(',')
            currfields = eles[0].convars

            # Construct a mapping between the solution file and the system
            try:
                smap = [solnfields.index(cf) for cf in currfields]
            except ValueError:
                raise RuntimeError('Invalid solution for system')

            # Process the solution
            for etype, ele in elemap.items():
                soln = initsoln[etype][:, smap, :]
                ele.set_ics_from_soln(soln, solncfg)
        else:
            for ele in eles:
                ele.set_ics_from_cfg()

        # Allocate these elements on the backend
        for etype, ele in elemap.items():
            curved = mesh.spts_curved[etype]
            linoff = np.max(*np.nonzero(curved), initial=-1) + 1

            ele.set_backend(self.backend, nregs, nonce, linoff)

        return eles, elemap

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

    def _load_bc_inters(self, mesh, elemap):
        bccls = self.bbcinterscls
        bcmap = {b.type: b for b in subclasses(bccls, just_leaf=True)}

        bc_inters = []
        for bname, interarr in mesh.bcon.items():
            # Determine the config file section
            cfgsect = f'soln-bcs-{bname}'

            # Instantiate
            bcclass = bcmap[self.cfg.get(cfgsect, 'type')]
            bciface = bcclass(self.backend, interarr, elemap, cfgsect,
                                self.cfg)
            bc_inters.append(bciface)

        return bc_inters

    def _gen_kernels(self, nregs, eles, iint, mpiint, bcint):
        self._kernels = kernels = defaultdict(list)

        # Helper function to tag the element type/MPI interface
        # associated with a kernel; used for dependency analysis
        self._ktags = {}

        def tag_kern(pname, prov, kern):
            if pname == 'eles':
                self._ktags[kern] = f'e-{prov.basis.name}'
            elif pname == 'mpiint':
                self._ktags[kern] = f'i-{prov.name}'

        provnames = ['eles', 'iint', 'mpiint', 'bcint']
        provlists = [eles, iint, mpiint, bcint]

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
                        kern = kgetter()
                        if isinstance(kern, NullKernel):
                            continue

                        kernels[f'{pn}/{kn}', None, None].append(kern)

                        tag_kern(pn, p, kern)

    def _gen_mpireqs(self, mpiint):
        self._mpireqs = mpireqs = defaultdict(list)

        for mn, mgetter in it.chain(*[m.mpireqs.items() for m in mpiint]):
            mpireqs[mn].append(mgetter())

    @memoize
    def _get_kernels(self, uinbank, foutbank):
        kernels = defaultdict(list)

        # Filter down the kernels dictionary
        for (kn, ui, fo), k in self._kernels.items():
            if ((ui is None and fo is None) or
                (ui is not None and ui == uinbank) or
                (fo is not None and fo == foutbank)):
                kernels[kn] = k

        # Obtain the bind method for kernels which take runtime arguments
        binders = [k.bind for k in it.chain(*kernels.values())
                   if hasattr(k, 'bind')]

        return kernels, binders

    def _kdeps(self, kdict, kern, *dnames):
        deps = []

        for name in dnames:
            for k in kdict[name]:
                if self._ktags[kern] == self._ktags[k]:
                    deps.append(k)

        return deps

    def _prepare_kernels(self, t, uinbank, foutbank):
        _, binders = self._get_kernels(uinbank, foutbank)

        for b in self._bc_inters:
            b.prepare(t)

        for b in binders:
            b(t=t)

    def _rhs_graphs(self, uinbank, foutbank):
        pass

    def rhs(self, t, uinbank, foutbank):
        self._rhs_uin_fout.add((uinbank, foutbank))
        self._prepare_kernels(t, uinbank, foutbank)

        for graph in self._rhs_graphs(uinbank, foutbank):
            self.backend.run_graph(graph)

    def _preproc_graphs(self, uinbank):
        pass

    def preproc(self, t, uinbank):
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

    def _compute_grads_graph(self, t, uinbank):
        raise NotImplementedError(f'Solver "{self.name}" does not compute '
                                  'corrected gradients of the solution')

    def compute_grads(self, t, uinbank):
        self._prepare_kernels(t, uinbank, None)

        for graph in self._compute_grads_graph(uinbank):
            self.backend.run_graph(graph)

    def filt(self, uinoutbank):
        kkey = ('eles/modal_filter', uinoutbank, None)

        self.backend.run_kernels(self._kernels[kkey])

    def evalsrcmacros(self, uinoutbank):
        kkey = ('eles/evalsrcmacros', uinoutbank, None)

        self.backend.run_kernels(self._kernels[kkey])

    def ele_scal_upts(self, idx):
        return [eb[idx].get() for eb in self.ele_banks]

    def get_ele_entmin_int(self):
        return [e.get() for e in self.eles_entmin_int]

    def _group(self, g, kerns, subs=[]):
        # Eliminate non-existent kernels
        kerns = [k for k in kerns if k is not None]

        # Eliminate substitutions associated with non-existent kernels
        subs = [[(k, n) for k, n in sub if k] for sub in subs]
        subs = [sub for sub in subs if len(sub) > 1]

        g.group(kerns, subs)

    def set_ele_entmin_int(self, entmin_int):
        for e, em in zip(self.eles_entmin_int, entmin_int):
            e.set(em)
