import math
import numpy as np

from pyfr.plugins.base import BaseSolverPlugin

class Source(BaseSolverPlugin):
    name = 'source'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)
        
        convars = intg.system.elementscls.convarmap[self.ndims]

        subs = self.cfg.items('constants')
        subs |= dict(x='ploc[0]', y='ploc[1]', z='ploc[2]')
        subs |= dict(abs='fabs', pi=math.pi)
        subs |= {v: f'u[{i}]' for i, v in enumerate(convars)}

        src_exprs = [self.cfg.getexpr('solver-plugin-source', v, '0', subs=subs)
                     for v in convars]

        if src_exprs:
            print('SOURCE')
            for etype, eles in intg.system.ele_map.items():
                eles.add_src_macro('pyfr.plugins.kernels.source','source', {'srcexprs': src_exprs}, ploc=True, soln=True)

    def __call__(self, intg):
        pass
