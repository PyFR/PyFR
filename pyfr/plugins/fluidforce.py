# -*- coding: utf-8 -*-

from collections import defaultdict
import os

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins.base import BasePlugin


class FluidForcePlugin(BasePlugin):
    name = 'fluidforce'
    systems = ['euler', 'navier-stokes']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Output frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Boundary to integrate over
        bc = 'bcon_{0}_p{1}'.format(suffix, intg.rallocs.prank)

        # Get the mesh and elements
        mesh, elemap = intg.system.mesh, intg.system.ele_map

        # See which ranks have the boundary
        bcranks = comm.gather(bc in mesh, root=root)

        # The root rank needs to open the output file
        if rank == root:
            if not any(bcranks):
                raise RuntimeError('Boundary {0} does not exist'
                                   .format(suffix))

            # Determine the file path
            fname = self.cfg.get(cfgsect, 'file')

            # Append the '.csv' extension
            if not fname.endswith('.csv'):
                fname += '.csv'

            # Open for appending
            self.outf = open(fname, 'a')

            # Output a header if required
            if (os.path.getsize(fname) == 0 and
                self.cfg.getbool(cfgsect, 'header', True)):
                print(','.join('txyz'[:self.ndims + 1]), file=self.outf)

        # Interpolation matrices and quadrature weights
        self._m0 = m0 = {}
        self._qwts = qwts = defaultdict(list)

        # If we have the boundary then process the interface
        if bc in mesh:
            # Element indices and associated face normals
            eidxs = defaultdict(list)
            norms = defaultdict(list)

            for etype, eidx, fidx, flags in mesh[bc].astype('U4,i4,i1,i1'):
                eles = elemap[etype]

                if (etype, fidx) not in m0:
                    facefpts = eles.basis.facefpts[fidx]

                    m0[etype, fidx] = eles.basis.m0[facefpts]
                    qwts[etype, fidx] = eles.basis.fpts_wts[facefpts]

                # Unit physical normals and their magnitudes (including |J|)
                npn = eles.get_norm_pnorms(eidx, fidx)
                mpn = eles.get_mag_pnorms(eidx, fidx)

                eidxs[etype, fidx].append(eidx)
                norms[etype, fidx].append(mpn[:, None]*npn)

            self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
            self._norms = {k: np.array(v) for k, v in norms.items()}

    def __call__(self, intg):
        # Return if no output is due
        if intg.nsteps % self.nsteps:
            return

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Solution matrices indexed by element type
        solns = dict(zip(intg.system.ele_types, intg.soln))

        # Force vector
        f = np.zeros(self.ndims)

        for etype, fidx in self._m0:
            # Get the interpolation operator
            m0 = self._m0[etype, fidx]
            nfpts, nupts = m0.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][..., self._eidxs[etype, fidx]]

            # Interpolate to the face
            ufpts = np.dot(m0, uupts.reshape(nupts, -1))
            ufpts = ufpts.reshape(nfpts, self.nvars, -1)
            ufpts = ufpts.swapaxes(0, 1)

            # Compute the pressure
            p = self.elementscls.conv_to_pri(ufpts, self.cfg)[-1]

            # Get the quadrature weights and normal vectors
            qwts = self._qwts[etype, fidx]
            norms = self._norms[etype, fidx]

            # Do the quadrature
            f += np.einsum('i...,ij,jik', qwts, p, norms)

        # Reduce and output if we're the root rank
        if rank != root:
            comm.Reduce(f, None, op=get_mpi('sum'), root=root)
        else:
            comm.Reduce(get_mpi('in_place'), f, op=get_mpi('sum'), root=root)

            # Build the row
            row = [intg.tcurr] + f.tolist()

            # Write
            print(','.join(str(r) for r in row), file=self.outf)

            # Flush to disk
            self.outf.flush()
