# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseSystem


class BaseAdvectionSystem(BaseSystem):
    _nqueues = 2

    def _get_negdivf(self):
        runall = self._backend.runall
        q1, q2 = self._queues
        kernels = self._kernels

        # Evaluate the solution at the flux points and pack up any
        # flux point solutions which are on our side of an MPI
        # interface
        q1 << kernels['eles', 'disu_fpts']()
        q1 << kernels['mpiint', 'scal_fpts_pack']()
        runall([q1])

        # Evaluate the flux at each of the solution points and take the
        # divergence of this to yield the transformed, partially
        # corrected, flux divergence.  Finally, solve the Riemann
        # problem at each interface to yield a common flux
        q1 << kernels['eles', 'tdisf_upts']()
        q1 << kernels['eles', 'tdivtpcorf_upts']()
        q1 << kernels['iint', 'comm_flux']()
        q1 << kernels['bcint', 'comm_flux']()

        # Send the MPI interface buffers we have just packed and
        # receive the corresponding buffers from our peers.  Then
        # proceed to unpack these received buffers
        q2 << kernels['mpiint', 'scal_fpts_send']()
        q2 << kernels['mpiint', 'scal_fpts_recv']()
        q2 << kernels['mpiint', 'scal_fpts_unpack']()

        runall([q1, q2])

        # Solve the remaining Riemann problems for the MPI interfaces
        # and use the complete set of common fluxes to generate the
        # fully corrected transformed flux divergence.  Finally,
        # negate and un-transform this divergence to give -∇·f.
        q1 << kernels['mpiint', 'comm_flux']()
        q1 << kernels['eles', 'tdivtconf_upts']()
        q1 << kernels['eles', 'negdivconf_upts']()
        runall([q1])
