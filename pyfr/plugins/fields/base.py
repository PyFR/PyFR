import re


class BaseFieldProvider:
    name = None
    export_types = None
    needs_grads = False
    deps = []
    fields = {}

    def __init__(self, ndims, cfg, export_type=None):
        self.ndims = ndims
        self.cfg = cfg

        if export_type is not None:
            if not re.fullmatch(self.export_types, export_type):
                raise RuntimeError(f'Field provider {self.name} does not '
                                   f'support {export_type} export')

    def run(self, view):
        if self.needs_grads and not view.has_grads:
            raise RuntimeError(f'Field provider {self.name} requires gradient '
                               'data in the solution')
        self._process(view)

    def _process(self, view):
        pass
