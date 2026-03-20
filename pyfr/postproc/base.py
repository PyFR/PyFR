from pyfr.plugins.base import BasePlugin


class BasePostProcPlugin(BasePlugin):
    prefix = 'postproc'
    export_types = None
    needs_gradients = False
    needs_normals = False

    def __init__(self, ndims, cfg, export_type=None):
        cfgsect = f'postproc-plugin-{self.name}'
        super().__init__(cfg, cfgsect, ndims)

        if export_type is not None:
            if not ('*' in self.export_types
                    or export_type in self.export_types):
                raise RuntimeError(f'Postproc {self.name} does not support '
                                   f'{export_type} export')

    def fields(self):
        return {}

    def compute(self, data):
        return {}
