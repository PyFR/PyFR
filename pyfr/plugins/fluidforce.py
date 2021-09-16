# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins.base import BasePlugin, init_csv


class FluidForcePlugin(BasePlugin):
    name = 'fluidforce'
    systems = ['ac-euler', 'ac-navier-stokes', 'euler', 'navier-stokes']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Output frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Check if we need to compute viscous force
        self._viscous = 'navier-stokes' in intg.system.name

        # Check if the system is incompressible
        self._ac = intg.system.name.startswith('ac')

        # Viscous correction
        self._viscorr = self.cfg.get('solver', 'viscosity-correction', 'none')

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Boundary to integrate over
        bc = f'bcon_{suffix}_p{intg.rallocs.prank}'

        # Moments
        mcomp = 3 if self.ndims == 3 else 1
        self._mcomp = mcomp if self.cfg.hasopt(cfgsect, 'morigin') else 0
        if self._mcomp:
            morigin = np.array(self.cfg.getliteral(cfgsect, 'morigin'))
            if len(morigin) != self.ndims:
                raise ValueError(f'morigin must have {self.ndims} components')

        # Get the mesh and elements
        mesh, elemap = intg.system.mesh, intg.system.ele_map

        # See which ranks have the boundary
        bcranks = comm.gather(bc in mesh, root=root)

        # The root rank needs to open the output file
        if rank == root:
            if not any(bcranks):
                raise RuntimeError(f'Boundary {suffix} does not exist')

            # CSV header
            header = ['t', 'px', 'py', 'pz'][:self.ndims + 1]
            if self._mcomp:
                header += ['mpx', 'mpy', 'mpz'][3 - mcomp:]
            if self._viscous:
                header += ['vx', 'vy', 'vz'][:self.ndims]
                if self._mcomp:
                    header += ['mvx', 'mvy', 'mvz'][3 - mcomp:]

            # Open
            self.outf = init_csv(self.cfg, cfgsect, ','.join(header))

        # Interpolation matrices and quadrature weights
        self._m0 = m0 = {}
        self._qwts = qwts = defaultdict(list)

        if self._viscous:
            self._m4 = m4 = {}
            rcpjact = {}

        # If we have the boundary then process the interface
        if bc in mesh:
            # Element indices, associated face normals and relative flux
            # points position with respect to the moments origin
            eidxs = defaultdict(list)
            norms = defaultdict(list)
            rfpts = defaultdict(list)

            for etype, eidx, fidx, flags in mesh[bc].astype('U4,i4,i1,i2'):
                eles = elemap[etype]

                if (etype, fidx) not in m0:
                    facefpts = eles.basis.facefpts[fidx]

                    m0[etype, fidx] = eles.basis.m0[facefpts]
                    qwts[etype, fidx] = eles.basis.fpts_wts[facefpts]

                if self._viscous and etype not in m4:
                    m4[etype] = eles.basis.m4

                    # Get the smats at the solution points
                    smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)

                    # Get |J|^-1 at the solution points
                    rcpdjac = eles.rcpdjac_at_np('upts')

                    # Product to give J^-T at the solution points
                    rcpjact[etype] = smat*rcpdjac

                # Unit physical normals and their magnitudes (including |J|)
                npn = eles.get_norm_pnorms(eidx, fidx)
                mpn = eles.get_mag_pnorms(eidx, fidx)

                eidxs[etype, fidx].append(eidx)
                norms[etype, fidx].append(mpn[:, None]*npn)

                # Get the flux points position of the given face and element
                # indices relative to the moment origin
                if self._mcomp:
                    fpts_idx = eles.basis.facefpts[fidx]
                    rfpt = eles.plocfpts[fpts_idx, eidx] - morigin
                    rfpts[etype, fidx].append(rfpt)

            self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
            self._norms = {k: np.array(v) for k, v in norms.items()}
            self._rfpts = {k: np.array(v) for k, v in rfpts.items()}

            if self._viscous:
                self._rcpjact = {k: rcpjact[k[0]][..., v]
                                 for k, v in self._eidxs.items()}

    def __call__(self, intg):
        # Return if no output is due
        if intg.nacptsteps % self.nsteps:
            return

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Solution matrices indexed by element type
        solns = dict(zip(intg.system.ele_types, intg.soln))
        ndims, nvars, mcomp = self.ndims, self.nvars, self._mcomp

        # Force and moment vectors
        fm = np.zeros((2 if self._viscous else 1, ndims + mcomp))

        for etype, fidx in self._m0:
            # Get the interpolation operator
            m0 = self._m0[etype, fidx]
            nfpts, nupts = m0.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][..., self._eidxs[etype, fidx]]

            # Interpolate to the face
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nvars, -1)
            ufpts = ufpts.swapaxes(0, 1)

            # Compute the pressure
            pidx = 0 if self._ac else -1
            p = self.elementscls.con_to_pri(ufpts, self.cfg)[pidx]

            # Get the quadrature weights and normal vectors
            qwts = self._qwts[etype, fidx]
            norms = self._norms[etype, fidx]

            # Do the quadrature
            fm[0, :ndims] += np.einsum('i...,ij,jik', qwts, p, norms)

            if self._viscous:
                # Get operator and J^-T matrix
                m4 = self._m4[etype]
                rcpjact = self._rcpjact[etype, fidx]

                # Transformed gradient at solution points
                tduupts = m4 @ uupts.reshape(nupts, -1)
                tduupts = tduupts.reshape(ndims, nupts, nvars, -1)

                # Physical gradient at solution points
                duupts = np.einsum('ijkl,jkml->ikml', rcpjact, tduupts)
                duupts = duupts.reshape(ndims, nupts, -1)

                # Interpolate gradient to flux points
                dufpts = np.array([m0 @ du for du in duupts])
                dufpts = dufpts.reshape(ndims, nfpts, nvars, -1)
                dufpts = dufpts.swapaxes(1, 2)

                # Viscous stress
                if self._ac:
                    vis = self.ac_stress_tensor(dufpts)
                else:
                    vis = self.stress_tensor(ufpts, dufpts)

                # Do the quadrature
                fm[1, :ndims] += np.einsum('i...,klij,jil', qwts, vis, norms)

            if self._mcomp:
                # Get the flux points positions relative to the moment origin
                rfpts = self._rfpts[etype, fidx]

                # Do the cross product with the normal vectors
                rcn = np.atleast_3d(np.cross(rfpts, norms))

                # Pressure force moments
                fm[0, ndims:] += np.einsum('i...,ij,jik->k', qwts, p, rcn)

                if self._viscous:
                    # Normal viscous force at each flux point
                    viscf = np.einsum('ijkl,lkj->lki', vis, norms)

                    # Normal viscous force moments at each flux point
                    rcf = np.atleast_3d(np.cross(rfpts, viscf))

                    # Do the quadrature
                    fm[1, ndims:] += np.einsum('i,jik->k', qwts, rcf)

        # Reduce and output if we're the root rank
        if rank != root:
            comm.Reduce(fm, None, op=get_mpi('sum'), root=root)
        else:
            comm.Reduce(get_mpi('in_place'), fm, op=get_mpi('sum'), root=root)

            # Write
            print(intg.tcurr, *fm.ravel(), sep=',', file=self.outf)

            # Flush to disk
            self.outf.flush()

    def stress_tensor(self, u, du):
        c = self._constants

        # Density, energy
        rho, E = u[0], u[-1]

        # Gradient of density and momentum
        gradrho, gradrhou = du[:, 0], du[:, 1:-1]

        # Gradient of velocity
        gradu = (gradrhou - gradrho[:, None]*u[None, 1:-1]/rho) / rho

        # Bulk tensor
        bulk = np.eye(self.ndims)[:, :, None, None]*np.trace(gradu)

        # Viscosity
        mu = c['mu']

        if self._viscorr == 'sutherland':
            cpT = c['gamma']*(E/rho - 0.5*np.sum(u[1:-1]**2, axis=0)/rho**2)
            Trat = cpT/c['cpTref']
            mu *= (c['cpTref'] + c['cpTs'])*Trat**1.5 / (cpT + c['cpTs'])

        return -mu*(gradu + gradu.swapaxes(0, 1) - 2/3*bulk)

    def ac_stress_tensor(self, du):
        # Gradient of velocity and kinematic viscosity
        gradu, nu = du[:, 1:], self._constants['nu']

        return -nu*(gradu + gradu.swapaxes(0, 1))
