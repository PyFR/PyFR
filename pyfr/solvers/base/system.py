# -*- coding: utf-8 -*-

from collections import defaultdict
import inspect
import itertools as it
import re

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

        if hasattr(eles[0], '_vect_upts'):
            self.eles_vect_upts = [e._vect_upts for e in eles]

        # Save the number of dimensions and field variables
        self.ndims = eles[0].ndims
        self.nvars = eles[0].nvars

        # Load the interfaces
        int_inters = self._load_int_inters(rallocs, mesh, elemap)
        mpi_inters = self._load_mpi_inters(rallocs, mesh, elemap)
        bc_inters = self._load_bc_inters(rallocs, mesh, elemap)
        backend.commit()

        # Queue
        self._queue = backend.queue()

        # Prepare the kernels and any associated MPI requests
        self._gen_kernels(nregs, eles, int_inters, mpi_inters, bc_inters)
        self._gen_mpireqs(mpi_inters)
        backend.commit()

        # Save the BC interfaces, but delete the memory-intensive elemap
        self._bc_inters = bc_inters
        for b in bc_inters:
            del b.elemap

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
            k = f'spt_{etype}_p{rallocs.prank}'

            try:
                curved = ~mesh[k, 'linear']
                linoff = np.max(*np.nonzero(curved), initial=-1) + 1
            except KeyError:
                linoff = ele.neles

            ele.set_backend(self.backend, nregs, nonce, linoff)

        return eles, elemap

    def _load_int_inters(self, rallocs, mesh, elemap):
        key = f'con_p{rallocs.prank}'

        lhs, rhs = mesh[key].astype('U4,i4,i1,i2').tolist()
        int_inters = self.intinterscls(self.backend, lhs, rhs, elemap,
                                       self.cfg)

        return [int_inters]

    def _load_mpi_inters(self, rallocs, mesh, elemap):
        lhsprank = rallocs.prank

        mpi_inters = []
        for rhsprank in rallocs.prankconn[lhsprank]:
            rhsmrank = rallocs.pmrankmap[rhsprank]
            interarr = mesh[f'con_p{lhsprank}p{rhsprank}']
            interarr = interarr.astype('U4,i4,i1,i2').tolist()

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
                interarr = mesh[f].astype('U4,i4,i1,i2').tolist()

                # Instantiate
                bcclass = bcmap[self.cfg.get(cfgsect, 'type')]
                bciface = bcclass(self.backend, interarr, elemap, cfgsect,
                                  self.cfg)
                bc_inters.append(bciface)

        return bc_inters

    def _gen_kernels(self, nregs, eles, iint, mpiint, bcint):
        self._kernels = kernels = defaultdict(list)

        provnames = ['eles', 'iint', 'mpiint', 'bcint']
        provlists = [eles, iint, mpiint, bcint]

        for pn, plst in zip(provnames, provlists):
            for kn, kgetter in it.chain(*[p.kernels.items() for p in plst]):
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
                else:
                    kern = kgetter()
                    if isinstance(kern, NullKernel):
                        continue

                    kernels[f'{pn}/{kn}', None, None].append(kern)

    def _gen_mpireqs(self, mpiint):
        self._mpireqs = mpireqs = defaultdict(list)

        for mn, mgetter in it.chain(*[m.mpireqs.items() for m in mpiint]):
            mpireqs[mn[:-4] + 'send_recv'].append(mgetter())

    @memoize
    def _get_kernels(self, uinbank, foutbank):
        kernels = defaultdict(list)

        # Filter down the kernels dictionary
        for (kn, ui, fo), k in self._kernels.items():
            if ((ui is None and fo is None) or
                (ui is not None and ui == uinbank) or
                (fo is not None and fo == foutbank)):
                kernels[kn] = k

        return kernels

    def rhs(self, t, uinbank, foutbank):
        pass

    def compute_grads(self, t, uinbank):
        raise NotImplementedError(f'Solver "{self.name}" does not compute '
                                  'corrected gradients of the solution')

    def filt(self, uinoutbank):
        kkey = ('eles/filter_soln', uinoutbank, None)

        self._queue.enqueue_and_run(self._kernels[kkey])

    def ele_scal_upts(self, idx):
        return [eb[idx].get() for eb in self.ele_banks]
