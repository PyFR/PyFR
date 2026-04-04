import itertools as it
import math

from pyfr.nputil import npeval
from pyfr.solvers.base import BaseInters


class BaseAdvectionIntInters(BaseInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, elemap, cfg)

        self.name = 'internal'

        # Store topology for system-level view creation
        self.lhs = lhs
        self.rhs = rhs

        # Compute the `optimal' permutation for our interface
        scal = {t: e._scal_fpts for t, e in elemap.items()}
        self._gen_perm(lhs, rhs, scal)

        # Generate the constant matrices
        self._pnorm_lhs = self._const_mat(lhs, 'get_pnorms_for_inters')

    def _gen_perm(self, lhs, rhs, scal):
        # Arbitrarily, take the permutation which results in an optimal
        # memory access pattern for the LHS of the interface
        self._perm = self._get_perm_for_field(lhs, scal)


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
    def preparefn(cls, bciface, mesh, elemap):
        pass

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
