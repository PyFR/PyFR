# -*- coding: utf-8 -*-

from collections import defaultdict, OrderedDict

from mpi4py import MPI

from pyfr.bases import BasisBase
from pyfr.elements import Elements
from pyfr.interfaces import InternalInterfaces, MPIInterfaces
from pyfr.util import proxylist, all_subclasses


class MeshPartition(object):
    def __init__(self, be, rallocs, mesh, nsubanks, cfg):
        self._be = be
        self._cfg = cfg
        self._nsubanks = nsubanks

        # Map from our MPI rank to our physical mesh rank
        prank = rallocs.mprankmap[MPI.COMM_WORLD.rank]

        # Load the elements and interfaces from the mesh
        self._load_eles(prank, mesh)
        self._load_int_inters(prank, mesh)
        self._load_mpi_inters(prank, rallocs, mesh)

        # Prepare the queues and kernels
        self._gen_queues()
        self._gen_kernels()

    def _load_eles(self, prank, mesh):
        # Automagically construct the bases map ('hex' => HexBasis &c)
        basiscls = {b.name: b for b in all_subclasses(BasisBase)}

        self._elemaps = OrderedDict()
        for k in basiscls.keys():
            mk = 'spt_%s_p%d' % (k, prank)
            if mk in mesh:
                ele = Elements(basiscls[k], mesh[mk], self._cfg)
                self._elemaps[k] = ele

                ele.set_ics()
                ele.set_backend(self._be, self._nsubanks)

        self._eles = proxylist(self._elemaps.values())

    def _load_int_inters(self, prank, mesh):
        lhs, rhs = mesh['con_p%d' % (prank)]
        int_inters = InternalInterfaces(self._be, lhs, rhs, self._elemaps,
                                        self._cfg)

        # Although we only have a single internal interface instance
        # we wrap it in a proxylist for consistency
        self._int_inters = proxylist([int_inters])

    def _load_mpi_inters(self, prank, rallocs, mesh):
        self._mpi_inters = proxylist([])
        for rhsprank in rallocs.prankconn[prank]:
            rhsmrank = rallocs.pmrankmap[rhsprank]
            interarr = mesh['con_p%dp%d' % (prank, rhsprank)]

            mpiiface = MPIInterfaces(self._be, interarr, rhsmrank,
                                     self._elemaps, self._cfg)
            self._mpi_inters.append(mpiiface)

    def _gen_queues(self):
        self._queues = [self._be.queue(), self._be.queue()]

    def _gen_kernels(self):
        eles = self._eles
        int_inters = self._int_inters
        mpi_inters = self._mpi_inters

        # Generate the kernels over each element type
        self._disu_fpts_kerns = eles.get_disu_fpts_kern()
        self._tdisf_upts_kerns = eles.get_tdisf_upts_kern()
        self._divtdisf_upts_kerns = eles.get_divtdisf_upts_kern()
        self._nrmtdisf_fpts_kerns = eles.get_nrmtdisf_fpts_kern()
        self._tdivtconf_upts_kerns = eles.get_tdivtconf_upts_kern()
        self._divconf_upts_kerns = eles.get_divconf_upts_kern()

        # Generate MPI sending/recving kernels over each MPI interface
        self._mpi_inters_pack_kerns = mpi_inters.get_pack_kern()
        self._mpi_inters_send_kerns = mpi_inters.get_send_pack_kern()
        self._mpi_inters_recv_kerns = mpi_inters.get_recv_pack_kern()
        self._mpi_inters_unpack_kerns = mpi_inters.get_unpack_kern()

        # Generate the Riemann solvers for internal and MPI interfaces
        self._int_inters_rsolve_kerns = int_inters.get_rsolve_kern()
        self._mpi_inters_rsolve_kerns = mpi_inters.get_rsolve_kern()

    def __call__(self, uinbank, foutbank):
        runall = self._be.runall
        q1, q2 = self._queues

        # Set the banks to use for each element type
        self._eles.scal_upts_inb.set_bank(uinbank)
        self._eles.scal_upts_outb.set_bank(foutbank)


        # Evaluate the solution at the flux points and pack up any
        # flux point solutions which are on our side of an MPI
        # interface
        q1 << self._disu_fpts_kerns
        q1 << self._mpi_inters_pack_kerns
        runall([q1])

        # Evaluate the flux at each of the solution points; and take
        # the of this flux and with at the flux points.  Finally, use
        # this to solve the Riemann problem at each of the internal
        # interfaces to yield a common interface flux
        q1 << self._tdisf_upts_kerns
        q1 << self._divtdisf_upts_kerns
        q1 << self._int_inters_rsolve_kerns

        # Send the MPI interface buffers we have just packed and
        # receive the corresponding buffers from our peers.  Then
        # proceed to unpack these received buffers
        q2 << self._mpi_inters_send_kerns
        q2 << self._mpi_inters_recv_kerns
        q2 << self._mpi_inters_unpack_kerns

        runall([q1, q2])

        # Now we have the solutions for the remote side of each MPI
        # interface we can solve the remaining Riemann problems.  These
        # common fluxes can then be used to correct the flux at each
        # of the solution points.  Finally, all that remains is to
        # suitably transform the flux.
        q1 << self._mpi_inters_rsolve_kerns
        q1 << self._nrmtdisf_fpts_kerns
        q1 << self._tdivtconf_upts_kerns
        q1 << self._divconf_upts_kerns
        runall([q1])

        # Wait for all ranks to finish
        MPI.COMM_WORLD.barrier()

    @property
    def ele_banks(self):
        return [list(b) for b in self._eles.scal_upts_inb]
