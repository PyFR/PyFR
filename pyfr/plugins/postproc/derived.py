from functools import cached_property
import re

from pyfr.plugins.common import get_elementscls
from pyfr.plugins.postproc.base import BasePostProcPlugin
from pyfr.stats import eval_algebraic, soln_exprs


class TableDerivedPostProc(BasePostProcPlugin):
    systems = '.*'
    dimensions = '2|3'
    export_types = '.*'

    def __init__(self, name, source, cfg, export_type=None, want=None):
        self.name = name

        super().__init__(source, cfg, export_type, want)

        self.elementscls = get_elementscls(cfg)
        self.derived, self.support = soln_exprs(self.elementscls, source.ndims,
                                                cfg, [name])

        # Restrict the outputs to any requested subset
        if want is not None:
            self.derived = {n: e for n, e in self.derived.items()
                            if n in want}

        self.fields = {n: [n] for n in self.derived}

        # Quantities using surface geometry are boundary-export only
        if self._on_boundary and export_type != 'boundary':
            raise RuntimeError(f'Postproc {name} is only available for '
                               'boundary exports')

    @property
    def _exprs(self):
        return (*self.derived.values(), *self.support.values())

    @property
    def _on_boundary(self):
        return any(re.search(r'\b(norm_[xyz]|boundary_dist)\b', e)
                   for e in self._exprs)

    @property
    def needs_grads(self):
        return any(re.search(r'\bgrad_', e) for e in self._exprs)

    @cached_property
    def _pnames(self):
        privars = self.elementscls.privars(self.ndims, self.cfg)
        words = {w for e in self._exprs for w in re.findall(r'\w+', e)}
        grads = sorted(w for w in words if w.startswith('grad_'))

        return [v for v in privars if v in words] + grads

    def _process(self, data):
        # Gradient symbols reference the primitive variable gradients
        ns = {v: data[v] for v in self._pnames}

        # Surface geometry symbols on boundary exports
        if self._on_boundary:
            ns |= dict(zip(('norm_x', 'norm_y', 'norm_z'), data.normals))
            ns['boundary_dist'] = data.boundary_dist

        out = eval_algebraic(self.derived, ns, self.support)
        data.fields |= out
