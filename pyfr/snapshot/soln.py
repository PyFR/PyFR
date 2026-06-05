from pyfr.snapshot.file import FileSnapshot


class SolnSnapshot(FileSnapshot):
    @property
    def pris_names(self):
        return list(self.elementscls.privars(self.ndims, self.cfg))

    def to_pris(self, interp_data):
        return self.elementscls.con_to_pri(interp_data, self.cfg)

    def to_grad_pris(self, interp_data, grad_interp):
        return self.elementscls.grad_con_to_pri(interp_data, grad_interp,
                                                self.cfg)

    def primitive_var_groups(self):
        return self.elementscls.visvars(self.ndims, self.cfg)
