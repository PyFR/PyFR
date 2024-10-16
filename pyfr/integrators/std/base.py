from pyfr.integrators.base import BaseIntegrator, _common_plugin_prop
from pyfr.integrators.base import BaseCommon


class BaseStdIntegrator(BaseCommon, BaseIntegrator):
    formulation = 'std'

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        super().__init__(backend, mesh, initsoln, cfg)

        # Sanity checks
        if self.controller_needs_errest and not self.stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

        # Determine the amount of temp storage required by this method
        self.nregs = self.stepper_nregs

        # Construct the relevant system
        self.system = systemcls(backend, mesh, initsoln, nregs=self.nregs,
                                cfg=cfg)

        # Event handlers for advance_to
        self.plugins = self._get_plugins(initsoln)

        # Commit the sytem
        self.system.commit()

        # Register index list and current index
        self._regidx = list(range(self.nregs))
        self._idxcurr = 0

        # Pre-process solution
        self.system.preproc(self.tcurr, self._idxcurr)

        # Global degree of freedom count
        self._gndofs = self._get_gndofs()

    @_common_plugin_prop('_curr_soln')
    def soln(self):
        self.system.postproc(self._idxcurr)
        return self.system.ele_scal_upts(self._idxcurr)

    @_common_plugin_prop('_curr_grad_soln')
    def grad_soln(self):
        self.system.postproc(self._idxcurr)
        self.system.compute_grads(self.tcurr, self._idxcurr)
        return [e.get() for e in self.system.eles_vect_upts]

    @_common_plugin_prop('_curr_dt_soln')
    def dt_soln(self):
        soln = self.soln

        self.system.rhs(self.tcurr, self._idxcurr, self._idxcurr)
        dt_soln = self.system.ele_scal_upts(self._idxcurr)

        # Reset current register with original contents
        for e, s in zip(self.system.ele_banks, soln):
            e[self._idxcurr].set(s)

        return dt_soln

    @property
    def controller_needs_errest(self):
        pass

    @property
    def entmin(self):
        return self.system.get_ele_entmin_int()

    @entmin.setter
    def entmin(self, value):
        self.system.set_ele_entmin_int(value)
