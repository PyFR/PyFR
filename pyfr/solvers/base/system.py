# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
import itertools as it
import re

from pyfr.inifile import Inifile
from pyfr.shapes import BaseShape
from pyfr.util import proxylist, subclasses


class BaseSystem(object, metaclass=ABCMeta):
    elementscls = None
    intinterscls = None
    mpiinterscls = None
    bbcinterscls = None

    # Number of queues to allocate
    _nqueues = None

    def __init__(self, backend, rallocs, mesh, initsoln, nreg, cfg):
        self.backend = backend
        self.mesh = mesh
        self.cfg = cfg

        # Load the elements
        eles, elemap = self._load_eles(rallocs, mesh, initsoln, nreg)
        backend.commit()

        # Retain the element map; this may be deleted by clients
        self.ele_map = elemap

        # Get the banks, types, num DOFs and shapes of the elements
        self.ele_banks = list(eles.scal_upts_inb)
        self.ele_types = list(elemap)
        self.ele_ndofs = [e.neles*e.nupts*e.nvars for e in eles]
        self.ele_shapes = [(e.nupts, e.nvars, e.neles) for e in eles]

        # Get all the solution point locations for the elements
        self.ele_ploc_upts = [e.ploc_at_np('upts') for e in eles]

        # I/O banks for the elements
        self.eles_scal_upts_inb = eles.scal_upts_inb
        self.eles_scal_upts_outb = eles.scal_upts_outb

        # Save the number of dimensions and field variables
        self.ndims = eles[0].ndims
        self.nvars = eles[0].nvars

        # Load the interfaces
        int_inters = self._load_int_inters(rallocs, mesh, elemap)
        mpi_inters = self._load_mpi_inters(rallocs, mesh, elemap)
        bc_inters = self._load_bc_inters(rallocs, mesh, elemap)
        backend.commit()

        # Prepare the queues and kernels
        self._gen_queues()
        self._gen_kernels(eles, int_inters, mpi_inters, bc_inters)
        backend.commit()

    def _load_eles(self, rallocs, mesh, initsoln, nreg):
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        # Look for and load each element type from the mesh
        elemap = OrderedDict()
        for f in mesh:
            m = re.match('spt_(.+?)_p%d$' % rallocs.prank, f)
            if m:
                # Element type
                t = m.group(1)

                elemap[t] = self.elementscls(basismap[t], mesh[f], self.cfg)

        # Construct a proxylist to simplify collective operations
        eles = proxylist(elemap.values())

        # Set the initial conditions either from a pyfrs file or from
        # explicit expressions in the config file
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
            for k, ele in elemap.items():
                soln = initsoln['soln_%s_p%d' % (k, rallocs.prank)]
                ele.set_ics_from_soln(soln, solncfg)
        else:
            eles.set_ics_from_cfg()

        # Allocate these elements on the backend
        eles.set_backend(self.backend, nreg)

        return eles, elemap

    def _load_int_inters(self, rallocs, mesh, elemap):
        key = 'con_p{0}'.format(rallocs.prank)

        lhs, rhs = mesh[key].astype('U4,i4,i1,i1').tolist()
        int_inters = self.intinterscls(self.backend, lhs, rhs, elemap,
                                       self.cfg)

        # Although we only have a single internal interfaces instance
        # we wrap it in a proxylist for consistency
        return proxylist([int_inters])

    def _load_mpi_inters(self, rallocs, mesh, elemap):
        lhsprank = rallocs.prank

        mpi_inters = proxylist([])
        for rhsprank in rallocs.prankconn[lhsprank]:
            rhsmrank = rallocs.pmrankmap[rhsprank]
            interarr = mesh['con_p%dp%d' % (lhsprank, rhsprank)]
            interarr = interarr.astype('U4,i4,i1,i1').tolist()

            mpiiface = self.mpiinterscls(self.backend, interarr, rhsmrank,
                                         rallocs, elemap, self.cfg)
            mpi_inters.append(mpiiface)

        return mpi_inters

    def _load_bc_inters(self, rallocs, mesh, elemap):
        bccls = self.bbcinterscls
        bcmap = {b.type: b for b in subclasses(bccls, just_leaf=True)}

        bc_inters = proxylist([])
        for f in mesh:
            m = re.match('bcon_(.+?)_p%d$' % rallocs.prank, f)
            if m:
                # Get the region name
                rgn = m.group(1)

                # Determine the config file section
                cfgsect = 'soln-bcs-%s' % rgn

                # Get the interface
                interarr = mesh[f].astype('U4,i4,i1,i1').tolist()

                # Instantiate
                bcclass = bcmap[self.cfg.get(cfgsect, 'type')]
                bciface = bcclass(self.backend, interarr, elemap, cfgsect,
                                  self.cfg)
                bc_inters.append(bciface)

        return bc_inters

    def _gen_queues(self):
        self._queues = [self.backend.queue() for i in range(self._nqueues)]

    def _gen_kernels(self, eles, iint, mpiint, bcint):
        self._kernels = kernels = defaultdict(proxylist)

        provnames = ['eles', 'iint', 'mpiint', 'bcint']
        provobjs = [eles, iint, mpiint, bcint]

        for pn, pobj in zip(provnames, provobjs):
            for kn, kgetter in it.chain(*pobj.kernels.items()):
                kernels[pn, kn].append(kgetter())

    @abstractmethod
    def rhs(self, t, uinbank, foutbank):
        pass

    def filt(self, uinoutbank):
        self.eles_scal_upts_inb.active = uinoutbank

        self._queues[0] % self._kernels['eles', 'filter_soln']()

    def ele_scal_upts(self, idx):
        return [eb[idx].get() for eb in self.ele_banks]
