from pyfr.snapshot.file import FileSnapshot


class StatsSnapshot(FileSnapshot):
    def __init__(self, mesh, soln):
        super().__init__(mesh, soln)

        # Stat files dont carry gradients
        self.has_grads = False
        self.grad_data = {}

        # Map each [soln-plugin-tavg] avg-X = expr to the index of avg-X in
        # data fields.  Lets users reference tavg primitives by their semantic
        # expression name (rho, u, ...) rather than the raw avg-X field name.
        self._tavg_indices = {}
        section = self.stats.get('tavg', 'cfg-section', None)
        if section is not None and section in self.cfg.sections():
            for k in self.cfg.items(section, prefix='avg-'):
                if k in self._data_field_names:
                    self._tavg_indices[self.cfg.get(section, k).strip()] = \
                        self._data_field_names.index(k)

    @property
    def pris_names(self):
        if self._tavg_indices:
            return list(self._tavg_indices)
        return list(self._data_field_names)

    def primitive_var_groups(self):
        return {name: (name,) for name in self.pris_names}

    def to_pris(self, interp_data):
        if self._tavg_indices:
            return [interp_data[i] for i in self._tavg_indices.values()]
        return list(interp_data)
