import numpy as np

from pyfr.integrators.base import BaseIntegrator, _common_plugin_prop


class BaseExplicitIntegrator(BaseIntegrator):
    formulation = 'explicit'

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        super().__init__(backend, mesh, initsoln, cfg)

        # Sanity checks
        if self.controller_needs_errest and not self.stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

        if cfg.get('solver', 'shock-capturing', 'none') == 'entropy-filter':
            if self.controller_needs_errest:
                raise TypeError('Entropy filtering not compatible with '
                                'error-estimation-based controllers')

        # Construct the relevant system
        self.system = systemcls(backend, mesh, initsoln, self._nregs, cfg,
                                self.serialiser,
                                needs_cfl=self.controller_needs_cfl)

        # Assign register numbers to our member variables
        self._assign_registers()

        # Event handlers for advance_to
        self.plugins = self._get_plugins(initsoln)

        # Commit the system
        self.system.commit()

        # Index of the register number containing the solution
        self.idxcurr = 0

        # Pre-process solution
        self.system.preproc(self.tcurr, self.idxcurr)

        # Global degree of freedom count
        self.gndofs = self._get_gndofs()

    @_common_plugin_prop('_curr_soln')
    def soln(self):
        self.system.postproc(self.idxcurr)
        return self.system.ele_scal_upts(self.idxcurr)

    @_common_plugin_prop('_curr_grad_soln')
    def grad_soln(self):
        self.system.postproc(self.idxcurr)
        self.compute_grads()
        return [e.get() for e in self.system.eles_vect_upts]

    @_common_plugin_prop('_curr_dt_soln')
    def dt_soln(self):
        soln = [np.require(s, requirements='O') for s in self.soln]

        self.system.rhs(self.tcurr, self.idxcurr, self.idxcurr)
        dt_soln = self.system.ele_scal_upts(self.idxcurr)

        # Reset current register with original contents
        for e, s in zip(self.system.ele_banks, soln):
            e[self.idxcurr].set(s)

        return dt_soln
