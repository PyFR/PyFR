# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
import re

from mpi4py import MPI

from pyfr.bases import BasisBase
from pyfr.elements import EulerElements, NavierStokesElements
from pyfr.inifile import Inifile
from pyfr.interfaces import (EulerIntInters, EulerMPIInters, EulerBaseBCInters,
                             NavierStokesIntInters, NavierStokesBaseBCInters,
                             NavierStokesMPIInters)
from pyfr.util import proxylist, subclass_map


def get_mesh_partition(backend, rallocs, mesh, initsoln, nreg, cfg):
    mpmap = subclass_map(BaseMeshPartition, 'name')
    mpcls = mpmap[cfg.get('solver', 'system')]

    return mpcls(backend, rallocs, mesh, initsoln, nreg, cfg)


class BaseMeshPartition(object):
    __metaclass__ = ABCMeta

    elementscls = None
    intinterscls = None
    mpiinterscls = None
    bbcinterscls = None

    def __init__(self, backend, rallocs, mesh, initsoln, nreg, cfg):
        self._backend = backend
        self._cfg = cfg
        self._nreg = nreg

        # Load the elements and interfaces from the mesh
        self._load_eles(rallocs, mesh, initsoln)
        self._load_int_inters(rallocs, mesh)
        self._load_mpi_inters(rallocs, mesh)
        self._load_bc_inters(rallocs, mesh)

        # Prepare the queues and kernels
        self._gen_queues()
        self._gen_kernels()

    def _load_eles(self, rallocs, mesh, initsoln):
        basismap = subclass_map(BasisBase, 'name')

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
            for k,ele in elemaps.items():
                soln = initsoln['soln_%s_p%d' % (k, rallocs.prank)]
                ele.set_ics_from_soln(soln, solncfg)
        else:
            eles.set_ics_from_expr()

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
            m = re.match('bcon_(.+?)_p%d' % rallocs.prank, f)
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
        self._queues = [self._backend.queue(), self._backend.queue()]

    @abstractmethod
    def _gen_kernels(self):
        eles = self._eles
        int_inters = self._int_inters
        mpi_inters = self._mpi_inters
        bc_inters = self._bc_inters

        # Generate the kernels over each element type
        self._disu_fpts_kerns = eles.get_disu_fpts_kern()
        self._tdisf_upts_kerns = eles.get_tdisf_upts_kern()
        self._tdivtpcorf_upts_kerns = eles.get_tdivtpcorf_upts_kern()
        self._tdivtconf_upts_kerns = eles.get_tdivtconf_upts_kern()
        self._negdivconf_upts_kerns = eles.get_negdivconf_upts_kern()

        # Generate MPI sending/recving kernels over each MPI interface
        self._mpi_inters_scal_fpts0_pack_kerns = \
            mpi_inters.get_scal_fpts0_pack_kern()
        self._mpi_inters_scal_fpts0_send_kerns = \
            mpi_inters.get_scal_fpts0_send_pack_kern()
        self._mpi_inters_scal_fpts0_recv_kerns = \
            mpi_inters.get_scal_fpts0_recv_pack_kern()
        self._mpi_inters_scal_fpts0_unpack_kerns = \
            mpi_inters.get_scal_fpts0_unpack_kern()

        # Generate the Riemann solvers for the various interface types
        self._int_inters_rsolve_kerns = int_inters.get_rsolve_kern()
        self._mpi_inters_rsolve_kerns = mpi_inters.get_rsolve_kern()
        self._bc_inters_rsolve_kerns = bc_inters.get_rsolve_kern()

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
        return list(self._elemaps.keys())

    @property
    def ele_shapes(self):
        return [(e.nupts, e.neles, e.nvars) for e in self._eles]

    @property
    def ele_ndofs(self):
        return [e.neles*e.nupts*e.nvars for e in self._eles]

    def ele_scal_upts(self, idx):
        return list(self._eles.get_scal_upts_mat(idx))


class BaseAdvectionMeshPartition(BaseMeshPartition):
    def _get_negdivf(self):
        runall = self._backend.runall
        q1, q2 = self._queues

        # Evaluate the solution at the flux points and pack up any
        # flux point solutions which are on our side of an MPI
        # interface
        q1 << self._disu_fpts_kerns()
        q1 << self._mpi_inters_scal_fpts0_pack_kerns()
        runall([q1])

        # Evaluate the flux at each of the solution points and take the
        # divergence of this to yield the transformed, partially
        # corrected, flux divergence.  Finally, solve the Riemann
        # problem at each interface to yield a common flux
        q1 << self._tdisf_upts_kerns()
        q1 << self._tdivtpcorf_upts_kerns()
        q1 << self._int_inters_rsolve_kerns()
        q1 << self._bc_inters_rsolve_kerns()

        # Send the MPI interface buffers we have just packed and
        # receive the corresponding buffers from our peers.  Then
        # proceed to unpack these received buffers
        q2 << self._mpi_inters_scal_fpts0_send_kerns()
        q2 << self._mpi_inters_scal_fpts0_recv_kerns()
        q2 << self._mpi_inters_scal_fpts0_unpack_kerns()

        runall([q1, q2])

        # Solve the remaining Riemann problems for the MPI interfaces
        # and use the complete set of common fluxes to generate the
        # fully corrected transformed flux divergence.  Finally,
        # negate and un-transform this divergence to give -∇·f.
        q1 << self._mpi_inters_rsolve_kerns()
        q1 << self._tdivtconf_upts_kerns()
        q1 << self._negdivconf_upts_kerns()
        runall([q1])


class EulerMeshPartition(BaseAdvectionMeshPartition):
    name = 'euler'

    elementscls = EulerElements
    intinterscls = EulerIntInters
    mpiinterscls = EulerMPIInters
    bbcinterscls = EulerBaseBCInters


class BaseAdvectionDiffusionMeshPartition(BaseAdvectionMeshPartition):
    def _gen_kernels(self):
        super(BaseAdvectionDiffusionMeshPartition, self)._gen_kernels()

        eles = self._eles
        int_inters = self._int_inters
        mpi_inters = self._mpi_inters
        bc_inters = self._bc_inters

        # Element-local kernels
        self._tgradpcoru_upts_kerns = eles.get_tgradpcoru_upts_kern()
        self._tgradcoru_upts_kerns = eles.get_tgradcoru_upts_kern()
        self._tgradcoru_fpts_kerns = eles.get_tgradcoru_fpts_kern()
        self._gradcoru_fpts_kerns = eles.get_gradcoru_fpts_kern()

        self._mpi_inters_vect_fpts0_pack_kerns = \
            mpi_inters.get_vect_fpts0_pack_kern()
        self._mpi_inters_vect_fpts0_send_kerns = \
            mpi_inters.get_vect_fpts0_send_pack_kern()
        self._mpi_inters_vect_fpts0_recv_kerns = \
            mpi_inters.get_vect_fpts0_recv_pack_kern()
        self._mpi_inters_vect_fpts0_unpack_kerns = \
            mpi_inters.get_vect_fpts0_unpack_kern()

        self._int_inters_conu_fpts_kerns = int_inters.get_conu_fpts_kern()
        self._mpi_inters_conu_fpts_kerns = mpi_inters.get_conu_fpts_kern()
        self._bc_inters_conu_fpts_kerns = bc_inters.get_conu_fpts_kern()

    def _get_negdivf(self):
        runall = self._backend.runall
        q1, q2 = self._queues

        q1 << self._disu_fpts_kerns()
        q1 << self._mpi_inters_scal_fpts0_pack_kerns()
        runall([q1])

        q1 << self._int_inters_conu_fpts_kerns()
        q1 << self._bc_inters_conu_fpts_kerns()
        q1 << self._tgradpcoru_upts_kerns()

        q2 << self._mpi_inters_scal_fpts0_send_kerns()
        q2 << self._mpi_inters_scal_fpts0_recv_kerns()
        q2 << self._mpi_inters_scal_fpts0_unpack_kerns()

        runall([q1, q2])

        q1 << self._mpi_inters_conu_fpts_kerns()
        q1 << self._tgradcoru_upts_kerns()
        q1 << self._tgradcoru_fpts_kerns()
        q1 << self._gradcoru_fpts_kerns()
        q1 << self._mpi_inters_vect_fpts0_pack_kerns()
        runall([q1])

        q1 << self._tdisf_upts_kerns()
        q1 << self._tdivtpcorf_upts_kerns()
        q1 << self._int_inters_rsolve_kerns()
        q1 << self._bc_inters_rsolve_kerns()

        q2 << self._mpi_inters_vect_fpts0_send_kerns()
        q2 << self._mpi_inters_vect_fpts0_recv_kerns()
        q2 << self._mpi_inters_vect_fpts0_unpack_kerns()

        runall([q1, q2])

        q1 << self._mpi_inters_rsolve_kerns()
        q1 << self._tdivtconf_upts_kerns()
        q1 << self._negdivconf_upts_kerns()
        runall([q1])


class NavierStokesMeshPartition(BaseAdvectionDiffusionMeshPartition):
    name = 'navier-stokes'

    elementscls = NavierStokesElements
    intinterscls = NavierStokesIntInters
    mpiinterscls = NavierStokesMPIInters
    bbcinterscls = NavierStokesBaseBCInters
