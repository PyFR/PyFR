# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseSystem
from pyfr.util import memoize


class BaseAdvectionSystem(BaseSystem):
    @memoize
    def _rhs_graphs(self, uinbank, foutbank):
        m = self._mpireqs
        k, _ = self._get_kernels(uinbank, foutbank)

        def edeps(ek, *edeps): return self._ele_deps(k, ek, *edeps)

        g1 = self.backend.graph()
        g1.add_all(k['eles/disu'])
        g1.add_all(k['mpiint/scal_fpts_pack'], deps=k['eles/disu'])
        g1.commit()

        g2 = self.backend.graph()
        g2.add_all(k['iint/comm_flux'])
        g2.add_all(k['bcint/comm_flux'])
        g2.add_all(k['eles/copy_soln'])
        g2.add_all(k['eles/qptsu'])
        for l in k['eles/tdisf_curved']:
            g2.add(l, deps=edeps(l, 'qptsu'))
        for l in k['eles/tdisf_linear']:
            g2.add(l, deps=edeps(l, 'qptsu'))
        for l in k['eles/tdivtpcorf']:
            g2.add(l, deps=edeps(l, 'tdisf_curved', 'tdisf_linear'))
        g2.commit()

        g3 = self.backend.graph()
        g3.add_all(k['mpiint/scal_fpts_unpack'])
        g3.add_all(k['mpiint/comm_flux'], deps=k['mpiint/scal_fpts_unpack'])
        g3.add_all(k['eles/tdivtconf'], deps=k['mpiint/comm_flux'])
        for l in k['eles/negdivconf']:
            g3.add(l, deps=edeps(l, 'tdivtconf'))
        g3.commit()

        return [(g1, None), (g2, m['scal_fpts_send_recv']), (g3, None)]
