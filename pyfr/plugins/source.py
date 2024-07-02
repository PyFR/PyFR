import math
import re

from pyfr.plugins.base import BaseSolverPlugin
from pyfr.util import first


class SourcePlugin(BaseSolverPlugin):
    name = 'source'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        convars = first(intg.system.ele_map.values()).convars

        subs = self.cfg.items('constants')
        subs |= dict(x='ploc[0]', y='ploc[1]', z='ploc[2]')
        subs |= dict(abs='fabs', pi=math.pi)
        subs |= {v: f'u[{i}]' for i, v in enumerate(convars)}

        src_exprs = [self.cfg.getexpr(cfgsect, v, subs=subs) for v in convars]

        ploc_in_src = any(re.search(r'\bploc\b', ex) for ex in src_exprs)
        soln_in_src = any(re.search(r'\bu\b', ex) for ex in src_exprs)

        for etype, eles in intg.system.ele_map.items():
            eles.add_src_macro('pyfr.plugins.kernels.source', 'source',
                               {'src_exprs': src_exprs}, ploc=ploc_in_src,
                               soln=soln_in_src)
