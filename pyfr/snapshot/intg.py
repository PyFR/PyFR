from pyfr.mpiutil import get_comm_rank_root
from pyfr.snapshot.base import BaseSnapshot


class IntgSnapshot(BaseSnapshot):
    prefix = 'soln'

    def __init__(self, intg):
        sys = intg.system

        self.mesh = sys.mesh
        self.cfg = intg.cfg
        self.elementscls = sys.elementscls
        self.ele_types = list(sys.ele_types)
        self.dtype = sys.backend.fpdtype
        self.has_grads = sys.eles_vect_upts is not None

        self._data_field_names = list(self.elementscls.convars(
            self.mesh.ndims, self.cfg))

        self._export_fields = sys.export_fields

        self.tcurr = intg.tcurr
        self.cycle = intg.nacptsteps

        self.data = {et: intg.soln[i] for i, et in enumerate(self.ele_types)}

        if self.has_grads:
            self.grad_data = {et: intg.grad_soln[i]
                              for i, et in enumerate(self.ele_types)}
        else:
            self.grad_data = {}

        comm, _, root = get_comm_rank_root()
        self.state = comm.bcast(intg.serialiser.serialise(), root=root)

    def aux(self, etype):
        return {ef.name: ef.getter()
                for ef in self._export_fields.get(etype, ())}

    def aux_info(self, etype):
        return {ef.name: (ef.shape, ef.dtype or self.dtype)
                for ef in self._export_fields.get(etype, ())}

    def to_pris(self, interp_data):
        return self.elementscls.con_to_pri(interp_data, self.cfg)

    def to_grad_pris(self, interp_data, grad_interp):
        return self.elementscls.grad_con_to_pri(interp_data, grad_interp,
                                                self.cfg)

    @property
    def pris_names(self):
        return list(self.elementscls.privars(self.ndims, self.cfg))

    def primitive_var_groups(self):
        return self.elementscls.visvars(self.ndims, self.cfg)
