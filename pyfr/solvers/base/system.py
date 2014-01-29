 # -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import re

from mpi4py import MPI

from pyfr.bases import BaseBasis
from pyfr.inifile import Inifile
from pyfr.util import proxylist, subclass_map


class BaseSystem(object):
    __metaclass__ = ABCMeta

    # Relevant derived classes
    elementscls = None
    intinterscls = None
    mpiinterscls = None
    bbcinterscls = None

    # Number of queues to allocate
    _nqueues = None

    def __init__(self, backend, rallocs, mesh, initsoln, nreg, cfg):
        self._backend = backend
        self._cfg = cfg
        self._nreg = nreg

        # Load the elements
        self._load_eles(rallocs, mesh, initsoln)
        backend.commit()

        # Load the interfaces
        self._load_int_inters(rallocs, mesh)
        self._load_mpi_inters(rallocs, mesh)
        self._load_bc_inters(rallocs, mesh)
        backend.commit()

        # Prepare the queues and kernels
        self._gen_queues()
        self._gen_kernels()

    def _load_eles(self, rallocs, mesh, initsoln):
        basismap = subclass_map(BaseBasis, 'name')

        # Look for and load each element type from the mesh
        self._elemaps = elemaps = OrderedDict()
        for bname, bcls in basismap.iteritems():
            mk = 'spt_%s_p%d' % (bname, rallocs.prank)
            if mk in mesh:
                elemaps[bname] = self.elementscls(bcls, mesh[mk], self._cfg)

        # Construct a proxylist to simplify collective operations
        self._eles = eles = proxylist(elemaps.values())

        # Set the initial conditions either from a pyfrs file or from
        # explicit expressions in the config file
        if initsoln:
            # Load the config used to produce the solution
            solncfg = Inifile(initsoln['config'].item())

            # Process the solution
            for k, ele in elemaps.iteritems():
                soln = initsoln['soln_%s_p%d' % (k, rallocs.prank)]
                ele.set_ics_from_soln(soln, solncfg)
        else:
            eles.set_ics_from_cfg()

        # Allocate these elements on the backend
        eles.set_backend(self._backend, self._nreg)

    def _load_int_inters(self, rallocs, mesh):
        lhs, rhs = mesh['con_p%d' % rallocs.prank]
        int_inters = self.intinterscls(self._backend, lhs, rhs, self._elemaps,
                                       self._cfg)

        # Although we only have a single internal interfaces instance
        # we wrap it in a proxylist for consistency
        self._int_inters = proxylist([int_inters])

    def _load_mpi_inters(self, rallocs, mesh):
        lhsprank = rallocs.prank

        self._mpi_inters = proxylist([])
        for rhsprank in rallocs.prankconn[lhsprank]:
            rhsmrank = rallocs.pmrankmap[rhsprank]
            interarr = mesh['con_p%dp%d' % (lhsprank, rhsprank)]

            mpiiface = self.mpiinterscls(self._backend, interarr, rhsmrank,
                                         rallocs, self._elemaps, self._cfg)
            self._mpi_inters.append(mpiiface)

    def _load_bc_inters(self, rallocs, mesh):
        bcmap = subclass_map(self.bbcinterscls, 'type')

        self._bc_inters = proxylist([])
        for f in mesh:
            m = re.match('bcon_(.+?)_p%d$' % rallocs.prank, f)
            if m:
                # Get the region name
                rgn = m.group(1)

                # Determine the config file section
                cfgsect = 'soln-bcs-%s' % rgn

                # Instantiate
                bcclass = bcmap[self._cfg.get(cfgsect, 'type')]
                bciface = bcclass(self._backend, mesh[f], self._elemaps,
                                  cfgsect, self._cfg)
                self._bc_inters.append(bciface)

    def _gen_queues(self):
        self._queues = [self._backend.queue() for i in xrange(self._nqueues)]

    @abstractmethod
    def _gen_kernels(self):
        pass

    @abstractmethod
    def _get_negdivf(self):
        pass

    def __call__(self, uinbank, foutbank):
        # Set the banks to use for each element type
        self._eles.scal_upts_inb.active = uinbank
        self._eles.scal_upts_outb.active = foutbank

        # Delegate to our subclass
        self._get_negdivf()

        # Wait for all ranks to finish
        MPI.COMM_WORLD.barrier()

    @property
    def ele_banks(self):
        return [list(b) for b in self._eles.scal_upts_inb]

    @property
    def ele_types(self):
        return list(self._elemaps)

    @property
    def ele_shapes(self):
        return [(e.nupts, e.nvars, e.neles) for e in self._eles]

    @property
    def ele_ndofs(self):
        return [e.neles*e.nupts*e.nvars for e in self._eles]

    def ele_scal_upts(self, idx):
        return list(self._eles.get_scal_upts_mat(idx))
