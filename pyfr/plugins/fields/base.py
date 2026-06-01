from graphlib import TopologicalSorter
import re

from pyfr.plugins.base import BasePlugin
from pyfr.util import subclasses


class BaseFieldProvider(BasePlugin):
    # Derived-field provider.  Reads pris/grad_pris/normals/etc from a
    # SampleView and writes a computed field back into view.fields.  Runs
    # anywhere a SnapshotSample is built — on-export (VTU), in-situ
    # (Catalyst/Ascent), live (sampler), offline (cli) — without caring how
    # the snapshot was sourced.  Providers read shared cfg sections directly
    # (`[constants]`, `[solver]`); no per-provider section exists.
    prefix = 'field'
    export_types = None
    needs_grads = False
    deps = []
    fields = {}

    def __init__(self, ndims, cfg, export_type=None):
        super().__init__(cfg=cfg, ndims=ndims)

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


def get_field_providers(names, ndims, cfg, export_type):
    available = {c.name: c for c in subclasses(BaseFieldProvider)}

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
