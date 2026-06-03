from pyfr.snapshot.file import FileSnapshot


class StatsSnapshot(FileSnapshot):
    def __init__(self, mesh, soln):
        super().__init__(mesh, soln)

        # Stat files dont carry gradients
        self.has_grads = False
        self.grad_data = None

    @property
    def pris_names(self):
        return list(self._data_field_names)

    def primitive_var_groups(self):
        # Identity mapping: each stored field is its own 1-component primitive.
        return {name: (name,) for name in self._data_field_names}

    def to_pris(self, interp_data, cfg):
        # Stored data is the primitive layout
        return list(interp_data)
