import re


class BaseFieldProvider:
    name = None
    export_types = None
    needs_grads = False
    kind = 'producer'
    deps = []
    fields = {}

    def __init__(self, ndims, cfg, export_type=None):
        self.ndims = ndims
        self.cfg = cfg

        pattern = self.export_types
        if export_type is not None and not re.fullmatch(pattern, export_type):
            raise RuntimeError(f'Field provider {self.name} does not '
                               f'support {export_type} export')

    def run(self, view):
        if self.needs_grads and not view.has_grads:
            raise RuntimeError(f'Field provider {self.name} requires gradient '
                               'data in the solution')
        self._process(view)

    def _process(self, view):
        pass
