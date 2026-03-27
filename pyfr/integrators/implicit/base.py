
from pyfr.integrators.base import BaseIntegrator, _common_plugin_prop, kernel_getter
from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll


class BaseImplicitIntegrator(BaseIntegrator):
    formulation = 'implicit'

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        super().__init__(backend, mesh, initsoln, cfg)

        # Sanity checks
        if self.controller_needs_errest and not self.stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

        if cfg.get('solver', 'shock-capturing', 'none') == 'entropy-filter':
            raise TypeError('Entropy filtering not compatible with '
                            'implicit time stepping')

        # Construct the relevant system
        self.system = systemcls(backend, mesh, initsoln, self._registers, cfg,
                                self.serialiser,
                                needs_cfl=self.controller_needs_cfl)

        # Assign register numbers to our member variables
        self._assign_registers()

        # Event handlers for advance_to
        self.plugins = self._get_plugins(initsoln)

        # Hook for subclasses to modify extents before commit
        self._pre_commit()

        # Commit the system
        self.system.commit()

        # Index of the register number containing the solution
        self.idxcurr = 0

        # Global degree of freedom count
        self.gndofs = self._get_gndofs()

    def _pre_commit(self):
        pass

    @_common_plugin_prop('_curr_soln')
    def soln(self):
        return self.system.ele_scal_upts(self.idxcurr)

    @_common_plugin_prop('_curr_grad_soln')
    def grad_soln(self):
        self.compute_grads()
        return [e.get() for e in self.system.eles_vect_upts]

    @_common_plugin_prop('_curr_dt_soln')
    def dt_soln(self):
        soln = self.soln

        self.system.rhs(self.tcurr, self.idxcurr, self.idxcurr)
        dt_soln = self.system.ele_scal_upts(self.idxcurr)

        # Reset current register with original contents
        for e, s in zip(self.system.ele_banks, soln):
            e[self.idxcurr].set(s)

        return dt_soln

    def _norm2(self, r, *, weights=(), norm_gndofs=False):
        comm, rank, root = get_comm_rank_root()

        # Run the kernels
        kerns = self._get_norm2_kerns(r, weights=tuple(weights))
        self.backend.run_kernels(kerns, wait=True)

        # Reduce over element types and ranks
        norm = scal_coll(comm.Allreduce, sum(k.retval[0] for k in kerns))
        scale = self.gndofs if norm_gndofs else 1

        return (norm / scale)**0.5

    @kernel_getter
    def _get_norm2_kerns(self, emats, x, *, weights=()):
        expr = ['(x/w)*(x/w)'] if weights else ['x*x']
        pvars = {'w': weights} if weights else {}
        return self.backend.kernel('reduction', 'sum', expr,
                                   {'x': emats[x]}, pvars=pvars)

    def _dot(self, a, b):
        comm, rank, root = get_comm_rank_root()
        kerns = self._get_dot_kerns(a, b)

        # Run the kernels
        self.backend.run_kernels(kerns, wait=True)

        # Reduce over element types and ranks and return
        return scal_coll(comm.Allreduce, sum(float(k.retval[0]) for k in kerns))

    @kernel_getter
    def _get_dot_kerns(self, emats, a, b):
        return self.backend.kernel('reduction', 'sum', ['a*b'],
                                   {'a': emats[a], 'b': emats[b]})

    def _multidot(self, a, b0, *bn):
        comm, rank, root = get_comm_rank_root()
        kerns = self._get_multidot_kerns(a, b0, *bn)

        self.backend.run_kernels(kerns, wait=True)

        # Reduce over element types and ranks and return
        results = sum([k.retval.astype(float) for k in kerns])
        comm.Allreduce(mpi.IN_PLACE, results)
        return results

    @kernel_getter
    def _get_multidot_kerns(self, emats, a, *bn):
        exprs = [f'a*b{i}' for i in range(len(bn))]
        vvars = {f'b{i}': emats[b] for i, b in enumerate(bn)}
        vvars['a'] = emats[a]
        return self.backend.kernel('reduction', 'sum', exprs, vvars)
