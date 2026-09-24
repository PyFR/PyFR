import itertools as it
import math

import numpy as np

from pyfr.exprs import npeval
from pyfr.nputil import batched_fuzzysort
from pyfr.solvers.base import BaseInters


class BaseAdvectionIntInters(BaseInters):
    # Number of elements per block when ordering the interface points
    blksz = 256

    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, elemap, cfg)

        self.name = 'internal'

        # Store topology for system-level view creation
        self.lhs = lhs
        self.rhs = rhs

        # Permute the RHS flux points so they pair with those of the LHS
        self._rhs_reorder = self._gen_rhs_reorder(lhs, rhs)

        # Compute the `optimal' permutation for our interface
        scal = {t: e._scal_fpts for t, e in elemap.items()}
        self._gen_perm(lhs, rhs, scal)

        # Generate the constant matrices
        self._pnorm_lhs = self._const_mat(lhs, 'get_pnorms_for_inters')

    def _gen_perm(self, lhs, rhs, scal):
        # Arbitrarily, take the permutation which results in an optimal
        # memory access pattern for the LHS of the interface
        self._perm = self._get_perm_for_field(lhs, scal, self.blksz)

    def _gen_rhs_reorder(self, lhs, rhs, tol=1e-6):
        # Faces which share a frame have their flux points paired already
        if rhs.transform is None:
            return None

        rot, shift = rhs.transform
        if not np.allclose(rot, np.eye(len(shift))):
            raise ValueError('Rotational periodicity is not supported')

        # Map the LHS flux points into the RHS frame
        lpts = self._inter_ploc(lhs) @ rot.T + shift
        rpts = self._inter_ploc(rhs)

        # Offset of each interface into the flux point arrays
        nfps = np.zeros(len(lhs), dtype=int)
        for etype, fidx, eidxs, idx in lhs.foreach():
            nfps[idx] = self.elemap[etype].nfacefpts[fidx]
        offs = np.concatenate(([0], np.cumsum(nfps[:-1])))

        reorder = np.empty(len(lpts), dtype=int)
        for nfp in np.unique(nfps):
            # Flux point indices of the interfaces with this many points
            ix = offs[nfps == nfp, None] + np.arange(nfp)

            # Sort both sides together so coincident points sort alike
            pts = np.concatenate([lpts[ix], rpts[ix]]).transpose(0, 2, 1)
            lperm, rperm = np.split(batched_fuzzysort(pts, tol), 2)

            # Pair each LHS flux point with the like-sorted RHS flux point
            rpos = np.take_along_axis(rperm, np.argsort(lperm, axis=1), 1)
            reorder[ix] = np.take_along_axis(ix, rpos, 1)

        # Ensure the paired flux points coincide
        if np.abs(lpts - rpts[reorder]).max() > tol:
            raise ValueError('Periodic flux points do not coincide')

        return reorder

    def side_perm(self, inter, with_perm=True):
        perm = super().side_perm(inter, with_perm)
        if inter is self.rhs and self._rhs_reorder is not None:
            return self._rhs_reorder[perm]
        else:
            return perm


class BaseAdvectionMPIInters(BaseInters):
    def __init__(self, be, lhs, rhsrank, elemap, cfg):
        super().__init__(be, lhs, elemap, cfg)
        self.rhsrank = rhsrank

        # Store topology for system-level view creation
        self.lhs = lhs

        # Name our interface so we can match kernels to MPI requests
        self.name = f'p{rhsrank}'

        # Per-interface MPI tag counter; all interfaces start at the
        # same base so that both sides of a connection always agree
        self._mpi_tag_counter = it.count()

        self._pnorm_lhs = self._const_mat(lhs, 'get_pnorms_for_inters')

    def next_mpi_tag(self):
        return next(self._mpi_tag_counter)


class BaseAdvectionBCInters(BaseInters):
    type = None

    @classmethod
    def common_consts(cls, cfg, cfgsect, ndims):
        return {}

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfg)

        # Store topology for system-level view creation
        self.lhs = lhs

        self.cfgsect = cfgsect
        self.bccomm = bccomm
        self.name = cfgsect.removeprefix('soln-bcs-')

        # For BC interfaces, which only have an LHS state, we take the
        # permutation which results in an optimal memory access pattern
        # iterating over this state.
        scal = {t: e._scal_fpts for t, e in elemap.items()}
        self._perm = self._get_perm_for_field(lhs, scal)

        # Constant matrices
        self._pnorm_lhs = self._const_mat(lhs, 'get_pnorms_for_inters')

        # Make the simulation time available inside kernels
        self.set_external('t', 'scalar fpdtype_t')

    @classmethod
    def hookfns(cls, bciface, mesh, elemap):
        return None, None

    def _eval_opts(self, opts, default=None):
        # Boundary conditions, much like initial conditions, can be
        # parameterized by values in [constants] so we must bring these
        # into scope when evaluating the boundary conditions
        cc = self.cfg.items_as('constants', float)

        cfg, sect = self.cfg, self.cfgsect

        # Evaluate any BC specific arguments from the config file
        if default is not None:
            return [npeval(cfg.getexpr(sect, k, default), cc) for k in opts]
        else:
            return [npeval(cfg.getexpr(sect, k), cc) for k in opts]

    def _exp_opts(self, opts, lhs, default={}):
        cfg, sect = self.cfg, self.cfgsect

        subs = cfg.items('constants')
        subs |= dict(x='ploc[0]', y='ploc[1]', z='ploc[2]')
        subs |= dict(abs='fabs', pi=str(math.pi))

        exprs = {}
        for k in opts:
            if k in default:
                exprs[k] = cfg.getexpr(sect, k, default[k], subs=subs)
            else:
                exprs[k] = cfg.getexpr(sect, k, subs=subs)

        if (any('ploc' in ex for ex in exprs.values()) and
            'ploc' not in self._external_args):
            spec = f'in fpdtype_t[{self.ndims}]'
            value = self._const_mat(lhs, 'get_ploc_for_inters')

            self.set_external('ploc', spec, value=value)

        return exprs
