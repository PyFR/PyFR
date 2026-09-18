import re

from pyfr.plugins.base import BasePlugin


class BasePostProcPlugin(BasePlugin):
    prefix = 'postproc'
    export_types = None
    needs_grads = False
    needs_gridh = False
    fields = {}

    # Transforms mutate the export data in place (coordinates, fields,
    # gradients) rather than deriving new named fields.  They declare no
    # `fields` and are run before any field-deriving plugin.
    transform = False

    # Data-source prefix this plugin attaches to (None = generic table
    # derivation).  A data source discovers its named plugins by this.
    source_prefix = None

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
