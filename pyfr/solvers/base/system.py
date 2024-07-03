from collections import defaultdict
import inspect
import itertools as it
import re
import statistics

import numpy as np

from pyfr.backends.base import NullKernel
from pyfr.inifile import Inifile
from pyfr.shapes import BaseShape
from pyfr.util import memoize, subclasses


class BaseSystem:
    elementscls = None
    intinterscls = None
    mpiinterscls = None
    bbcinterscls = None

    # Nonce sequence
    _nonce_seq = it.count()

    def __init__(self, backend, rallocs, mesh, initsoln, nregs, cfg):
        self.backend = backend
        self.mesh = mesh
        self.cfg = cfg
        self.nregs = nregs

        # Obtain a nonce to uniquely identify this system
        nonce = str(next(self._nonce_seq))

        # Load the elements
        eles, elemap = self._load_eles(rallocs, mesh, initsoln, nregs, nonce)
        backend.commit()

        # Retain the element map; this may be deleted by clients
        self.ele_map = elemap

        # Get the banks, types, num DOFs and shapes of the elements
        self.ele_banks = [e.scal_upts for e in eles]
        self.ele_types = list(elemap)
        self.ele_ndofs = [e.neles*e.nupts*e.nvars for e in eles]
        self.ele_shapes = [(e.nupts, e.nvars, e.neles) for e in eles]

        # Get all the solution point locations for the elements
        self.ele_ploc_upts = [e.ploc_at_np('upts') for e in eles]

        if hasattr(eles[0], '_grad_upts'):
            self.eles_vect_upts = [e._grad_upts for e in eles]

        if hasattr(eles[0], 'entmin_int'):
            self.eles_entmin_int = [e.entmin_int for e in eles]

        # Save the number of dimensions and field variables
        self.ndims = eles[0].ndims
        self.nvars = eles[0].nvars

        # Load the interfaces
        self._int_inters = self._load_int_inters(rallocs, mesh, elemap)
        self._mpi_inters = self._load_mpi_inters(rallocs, mesh, elemap)
        self._bc_inters = self._load_bc_inters(rallocs, mesh, elemap)
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

    def _load_eles(self, rallocs, mesh, initsoln, nregs, nonce):
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        # Look for and load each element type from the mesh
        elemap = {}
        for f in mesh:
            if (m := re.match(f'spt_(.+?)_p{rallocs.prank}$', f)):
                # Element type
                t = m[1]

                elemap[t] = self.elementscls(basismap[t], mesh[f], self.cfg)

        eles = list(elemap.values())

        # Set the initial conditions
        if initsoln:
            # Load the config and stats files from the solution
            solncfg = Inifile(initsoln['config'])
            solnsts = Inifile(initsoln['stats'])

            # Get the names of the conserved variables (fields)
            solnfields = solnsts.get('data', 'fields', '')
            currfields = ','.join(eles[0].convarmap[eles[0].ndims])

            # Ensure they match up
            if solnfields and solnfields != currfields:
                raise RuntimeError('Invalid solution for system')

            # Process the solution
            for etype, ele in elemap.items():
                soln = initsoln[f'soln_{etype}_p{rallocs.prank}']
                ele.set_ics_from_soln(soln, solncfg)
        else:
            for ele in eles:
                ele.set_ics_from_cfg()

        # Allocate these elements on the backend
        for etype, ele in elemap.items():
            curved = ~mesh[f'spt_{etype}_p{rallocs.prank}', 'linear']
            linoff = np.max(*np.nonzero(curved), initial=-1) + 1

            ele.set_backend(self.backend, nregs, nonce, linoff)

        return eles, elemap

    def _load_int_inters(self, rallocs, mesh, elemap):
        key = f'con_p{rallocs.prank}'

        lhs, rhs = mesh[key].tolist()
        int_inters = self.intinterscls(self.backend, lhs, rhs, elemap,
                                       self.cfg)

        return [int_inters]

    def _load_mpi_inters(self, rallocs, mesh, elemap):
        lhsprank = rallocs.prank

        mpi_inters = []
        for rhsprank in rallocs.prankconn[lhsprank]:
            rhsmrank = rallocs.pmrankmap[rhsprank]
            interarr = mesh[f'con_p{lhsprank}p{rhsprank}']
            interarr = interarr.tolist()

            mpiiface = self.mpiinterscls(self.backend, interarr, rhsmrank,
                                         rallocs, elemap, self.cfg)
            mpi_inters.append(mpiiface)

        return mpi_inters

    def _load_bc_inters(self, rallocs, mesh, elemap):
        bccls = self.bbcinterscls
        bcmap = {b.type: b for b in subclasses(bccls, just_leaf=True)}

        bc_inters = []
        for f in mesh:
            if (m := re.match(f'bcon_(.+?)_p{rallocs.prank}$', f)):
                # Determine the config file section
                cfgsect = f'soln-bcs-{m[1]}'

                # Get the interface
                interarr = mesh[f].tolist()

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
        # Eliminate non-existing kernels
        kerns = [k for k in kerns if k is not None]
        subs = [sub for sub in subs if None not in it.chain(*sub)]

        g.group(kerns, subs)

    def set_ele_entmin_int(self, entmin_int):
        for e, em in zip(self.eles_entmin_int, entmin_int):
            e.set(em)
