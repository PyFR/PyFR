import re

from pyfr.plugins.base import BasePlugin


class BasePostProcPlugin(BasePlugin):
    prefix = 'postproc'
    export_types = None
    needs_grads = False
    needs_gridh = False
    fields = {}

    def __init__(self, source, cfg, export_type=None, want=None):
        cfgsect = f'postproc-plugin-{self.name}'
        super().__init__(cfg=cfg, cfgsect=cfgsect, ndims=source.ndims)

        self.source = source

        if export_type is not None:
            if not re.fullmatch(self.export_types, export_type):
                raise RuntimeError(f'Postproc {self.name} does not support '
                                   f'{export_type} export')

    def derived_fields(self, fields):
        return {}

    def run(self, data):
        if self.needs_grads and not data.has_grads:
            raise RuntimeError(f'Postproc {self.name} requires gradient '
                               'data in the solution')
        self._process(data)

    def _process(self, data):
        pass
