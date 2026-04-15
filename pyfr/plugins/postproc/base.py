from graphlib import TopologicalSorter
import re

from pyfr.plugins.base import BasePlugin
from pyfr.util import subclass_where


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


def resolve_pp_plugins(names, ndims, cfg, export_type):
    # Gather requested plugins and their transitive dependencies
    classes = {}

    def visit(name):
        if name in classes:
            return
        cls = subclass_where(BasePostProcPlugin, name=name)
        classes[name] = cls
        for d in cls.deps:
            visit(d)

    for name in names:
        visit(name)

    graph = {n: set(c.deps) for n, c in classes.items()}
    ordered = TopologicalSorter(graph).static_order()

    return [classes[n](ndims, cfg, export_type) for n in ordered]
