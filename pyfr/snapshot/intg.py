from functools import cached_property

from pyfr.snapshot.base import BaseSnapshot


class IntgSnapshot(BaseSnapshot):
    prefix = 'soln'

    def __init__(self, intg):
        sys = intg.system

        self.mesh = sys.mesh
        self.config = intg.cfg
        self.elementscls = sys.elementscls
        self.ele_types = list(sys.ele_types)
        self.dtype = sys.backend.fpdtype
        self.has_grads = True

        self._data_field_names = list(self.elementscls.convars(
            self.mesh.ndims, self.config))

        self._export_fields = sys.export_fields
        self._intg = intg

    @property
    def tcurr(self):
        return self._intg.tcurr

    @property
    def cycle(self):
        return self._intg.nacptsteps

    @cached_property
    def data(self):
        return {et: self._intg.soln[i] for i, et in enumerate(self.ele_types)}

    @cached_property
    def grad_data(self):
        if not self.has_grads:
            return None
        return {et: self._intg.grad_soln[i]
                for i, et in enumerate(self.ele_types)}

    def aux(self, etype):
        return {ef.name: ef.getter()
                for ef in self._export_fields.get(etype, ())}

    def aux_info(self, etype):
        return {ef.name: (ef.shape, ef.dtype or self.dtype)
                for ef in self._export_fields.get(etype, ())}

    def to_pris(self, interp_data, cfg):
        return list(self.elementscls.con_to_pri(interp_data, cfg))

    def to_grad_pris(self, interp_data, grad_interp, cfg):
        return list(self.elementscls.grad_con_to_pri(interp_data, grad_interp,
                                                     cfg))

    @property
    def pris_names(self):
        return list(self.elementscls.privars(self.ndims, self.config))

    def primitive_var_groups(self):
        return self.elementscls.visvars(self.ndims, self.config)

    def compute_grads(self):
        if self.has_grads:
            self._intg.compute_grads()
