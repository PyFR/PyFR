from graphlib import TopologicalSorter
import re

from pyfr.plugins.base import BasePlugin
from pyfr.util import subclasses


class BasePostProcPlugin(BasePlugin):
    prefix = 'postproc'
    export_types = None
    needs_grads = False
    deps = []

    def __init__(self, ndims, cfg, export_type=None):
        cfgsect = f'postproc-plugin-{self.name}'
        super().__init__(cfg=cfg, cfgsect=cfgsect, ndims=ndims)

        if export_type is not None:
            if not re.fullmatch(self.export_types, export_type):
                raise RuntimeError(f'Postproc {self.name} does not support '
                                   f'{export_type} export')

    def fields(self):
        return {}

    def run(self, data):
        if self.needs_grads and not data.has_grads:
            raise RuntimeError(f'Postproc {self.name} requires gradient '
                               'data in the solution')
        self._process(data)

    def _process(self, data):
        pass


def get_pp_plugins(names, ndims, cfg, export_type):
    available = {c.name: c for c in subclasses(BasePostProcPlugin)}

    ts = TopologicalSorter()
    todo, added = list(names), set()

    while todo:
        if (name := todo.pop()) in added:
            continue

        deps = available[name].deps
        ts.add(name, *deps)
        todo.extend(deps)
        added.add(name)

    return [available[n](ndims, cfg, export_type) for n in ts.static_order()]
