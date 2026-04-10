import re

from pyfr.plugins.base import BasePlugin


class BasePostProcPlugin(BasePlugin):
    prefix = 'postproc'
    export_types = None
    needs_grads = False

    def __init__(self, ndims, cfg, export_type=None):
        cfgsect = f'postproc-plugin-{self.name}'
        super().__init__(cfg=cfg, cfgsect=cfgsect, ndims=ndims)

        if export_type is not None:
            if not re.fullmatch(self.export_types, export_type):
                raise RuntimeError(f'Postproc {self.name} does not support '
                                   f'{export_type} export')

    def fields(self):
        return {}

    def process(self, data):
        pass
