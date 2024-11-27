import itertools as it
import math

from pyfr.solvers.base import BaseInters
from pyfr.nputil import npeval


class BaseAdvectionIntersMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ef_enabled = (self.cfg.get('solver', 'shock-capturing') == 
                            'entropy-filter' and
                            self.cfg.getint('solver', 'order'))


class BaseAdvectionIntInters(BaseAdvectionIntersMixin, BaseInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, elemap, cfg)

        # Compute the `optimal' permutation for our interface
        self._gen_perm(lhs, rhs)

        # Generate the left and right hand side view matrices
        self._scal_lhs = self._scal_view(lhs, 'get_scal_fpts_for_inter')
        self._scal_rhs = self._scal_view(rhs, 'get_scal_fpts_for_inter')

        # Generate the additional view matrices for entropy filtering
        if self._ef_enabled:
            self._entmin_lhs = self._view(
                lhs, 'get_entmin_int_fpts_for_inter', with_perm=False
            )
            self._entmin_rhs = self._view(
                rhs, 'get_entmin_int_fpts_for_inter', with_perm=False
            )
        else:
            self._entmin_lhs = self._entmin_rhs = None

        # Generate the constant matrices
        self._pnorm_lhs = self._const_mat(lhs, 'get_pnorms_for_inter')

    def _gen_perm(self, lhs, rhs):
        # Arbitrarily, take the permutation which results in an optimal
        # memory access pattern for the LHS of the interface
        self._perm = self._get_perm_for_view(lhs, 'get_scal_fpts_for_inter')


class BaseAdvectionMPIInters(BaseAdvectionIntersMixin, BaseInters):
    # Starting tag used for MPI
    BASE_MPI_TAG = 2314

    def __init__(self, be, lhs, rhsrank, elemap, cfg):
        super().__init__(be, lhs, elemap, cfg)
        self._rhsrank = rhsrank

        # Name our interface so we can match kernels to MPI requests
        self.name = 'p{rhsrank}'

        # MPI request tag counter
        self._mpi_tag_counter = it.count(self.BASE_MPI_TAG)

        # Generate the left hand view matrix and its dual
        self._scal_lhs = self._scal_xchg_view(lhs, 'get_scal_fpts_for_inter')
        self._scal_rhs = be.xchg_matrix_for_view(self._scal_lhs)

        self._pnorm_lhs = self._const_mat(lhs, 'get_pnorms_for_inter')

        # Kernels
        self.kernels['scal_fpts_pack'] = lambda: be.kernel(
            'pack', self._scal_lhs
        )
        self.kernels['scal_fpts_unpack'] = lambda: be.kernel(
            'unpack', self._scal_rhs
        )

        # Associated MPI requests
        scal_fpts_tag = next(self._mpi_tag_counter)
        self.mpireqs['scal_fpts_send'] = lambda: self._scal_lhs.sendreq(
            self._rhsrank, scal_fpts_tag
        )
        self.mpireqs['scal_fpts_recv'] = lambda: self._scal_rhs.recvreq(
            self._rhsrank, scal_fpts_tag
        )

        if self._ef_enabled:
            self._entmin_lhs = self._xchg_view(
                lhs, 'get_entmin_int_fpts_for_inter', with_perm=False
            )
            self._entmin_rhs = be.xchg_matrix_for_view(self._entmin_lhs)

            self.kernels['ent_fpts_pack'] = lambda: be.kernel(
                'pack', self._entmin_lhs
            )
            self.kernels['ent_fpts_unpack'] = lambda: be.kernel(
                'unpack', self._entmin_rhs
            )

            ent_fpts_tag = next(self._mpi_tag_counter)
            self.mpireqs['ent_fpts_send'] = lambda: self._entmin_lhs.sendreq(
                self._rhsrank, ent_fpts_tag
            )
            self.mpireqs['ent_fpts_recv'] = lambda: self._entmin_rhs.recvreq(
                self._rhsrank, ent_fpts_tag
            )
        else:
            self._entmin_lhs = self._entmin_rhs = None


class BaseAdvectionBCInters(BaseAdvectionIntersMixin, BaseInters):
    type = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfg)
        self.cfgsect = cfgsect

        # For BC interfaces, which only have an LHS state, we take the
        # permutation which results in an optimal memory access pattern
        # iterating over this state.
        self._perm = self._get_perm_for_view(lhs, 'get_scal_fpts_for_inter')

        # LHS view and constant matrices
        self._scal_lhs = self._scal_view(lhs, 'get_scal_fpts_for_inter')
        self._pnorm_lhs = self._const_mat(lhs, 'get_pnorms_for_inter')

        # Make the simulation time available inside kernels
        self._set_external('t', 'scalar fpdtype_t')

        if self._ef_enabled:
            self._entmin_lhs = self._view(lhs, 'get_entmin_bc_fpts_for_inter')
        else:
            self._entmin_lhs = None

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
            value = self._const_mat(lhs, 'get_ploc_for_inter')

            self._set_external('ploc', spec, value=value)

        return exprs
