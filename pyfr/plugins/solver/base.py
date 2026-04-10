from pyfr.plugins.base import BasePlugin
from pyfr.plugins.mixins import InSituMixin


class BaseSolverPlugin(InSituMixin, BasePlugin):
    prefix = 'solver'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Runtime extern binding
        self._extern_values = {}
        self._extern_binders = []

    def _update_extern_values(self):
        pass

    def _register_externs(self, intg, names, spec='scalar fpdtype_t'):
        self._update_extern_values()

        for eles in intg.system.ele_map.values():
            for name in names:
                eles.set_external(name, spec)

        intg.system.register_kernel_callback(names, self._extern_callback)

    def _extern_callback(self, kern):
        self._extern_binders.append(kern.bind)
        kern.bind(**self._extern_values)

    def bind_externs(self):
        for b in self._extern_binders:
            b(**self._extern_values)
