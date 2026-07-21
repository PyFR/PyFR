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

    def _process(self, data):
        privars = self.elementscls.privars(self.ndims, self.cfg)
        ns = dict(zip(privars, data.pris))

        # Gradient symbols reference the primitive variable gradients
        if self.needs_grads:
            for v, dv in zip(privars, data.grad_pris):
                for x, dvx in zip('xyz', dv):
                    ns[f'grad_{v}_{x}'] = dvx

        # Surface geometry symbols on boundary exports
        if self._on_boundary:
            ns |= dict(zip(('norm_x', 'norm_y', 'norm_z'), data.normals))
            ns['boundary_dist'] = data.boundary_dist

        out = eval_algebraic(self.derived, ns, self.support)
        data.fields |= out
