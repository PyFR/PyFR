from pyfr.integrators.base import BaseIntegrator
from pyfr.integrators.base import BaseCommon


class BaseStdIntegrator(BaseCommon, BaseIntegrator):
    formulation = 'std'

    def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
        super().__init__(backend, rallocs, mesh, initsoln, cfg)

        # Sanity checks
        if self.controller_needs_errest and not self.stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

        # Determine the amount of temp storage required by this method
        self.nregs = self.stepper_nregs

        # Construct the relevant system
        self.system = systemcls(backend, rallocs, mesh, initsoln,
                                nregs=self.nregs, cfg=cfg)

        # Register index list and current index
        self._regidx = list(range(self.nregs))
        self._idxcurr = 0

        # Pre-process solution if necessary
        self.system.preproc(self.system.ele_scal_upts(self._idxcurr))

        # Global degree of freedom count
        self._gndofs = self._get_gndofs()

        # Event handlers for advance_to
        self.completed_step_handlers = self._get_plugins(initsoln)

        # Delete the memory-intensive elements map from the system
        del self.system.ele_map

    @property
    def soln(self):
        if not self._curr_soln:
            self.system.postproc(self._idxcurr)
            self._curr_soln = self.system.ele_scal_upts(self._idxcurr)

        return self._curr_soln

    @property
    def grad_soln(self):
        system = self.system

        if not self._curr_grad_soln:
            system.postproc(self._idxcurr)
            system.compute_grads(self.tcurr, self._idxcurr)
            self._curr_grad_soln = [e.get() for e in system.eles_vect_upts]

        return self._curr_grad_soln

    @property
    def controller_needs_errest(self):
        pass

    @property
    def entmin(self):
        return self.system.get_ele_entmin_int()

    @entmin.setter
    def entmin(self, value):
        self.system.set_ele_entmin_int(value)
