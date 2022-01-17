# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionSystem
from pyfr.util import memoize


class BaseAdvectionDiffusionSystem(BaseAdvectionSystem):
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
        g2.add_all(k['eles/copy_soln'])
        g2.add_all(k['iint/copy_fpts'])
        g2.add_all(k['iint/con_u'], deps=k['iint/copy_fpts'])
        g2.add_all(k['bcint/con_u'], deps=k['iint/copy_fpts'])
        g2.add_all(k['eles/shocksensor'])
        g2.add_all(k['mpiint/artvisc_fpts_pack'], deps=k['eles/shocksensor'])
        g2.add_all(k['eles/tgradpcoru_upts'])
        g2.commit()

        g3 = self.backend.graph()
        g3.add_all(k['mpiint/scal_fpts_unpack'])
        g3.add_all(k['mpiint/con_u'], deps=k['mpiint/scal_fpts_unpack'])
        g3.add_all(k['eles/tgradcoru_upts'], deps=k['mpiint/con_u'])
        for l in k['eles/gradcoru_upts_curved']:
            g3.add(l, deps=edeps(l, 'tgradcoru_upts'))
        for l in k['eles/gradcoru_upts_linear']:
            g3.add(l, deps=edeps(l, 'tgradcoru_upts'))
        for l in k['eles/gradcoru_fpts']:
            deps = edeps(l, 'gradcoru_upts_curved', 'gradcoru_upts_linear')
            g3.add(l, deps=deps)
        g3.add_all(k['mpiint/vect_fpts_pack'], deps=k['eles/gradcoru_fpts'])
        g3.commit()

        g4 = self.backend.graph()
        g4.add_all(k['mpiint/artvisc_fpts_unpack'])
        g4.add_all(k['iint/comm_flux'])
        g4.add_all(k['bcint/comm_flux'])
        g4.add_all(k['eles/gradcoru_qpts'])
        g4.add_all(k['eles/qptsu'])
        for l in k['eles/tdisf_curved']:
            g4.add(l, deps=edeps(l, 'gradcoru_qpts', 'qptsu'))
        for l in k['eles/tdisf_linear']:
            g4.add(l, deps=edeps(l, 'gradcoru_qpts', 'qptsu'))
        for l in k['eles/tdivtpcorf']:
            g4.add(l, deps=edeps(l, 'tdisf_curved', 'tdisf_linear'))
        g4.commit()

        g5 = self.backend.graph()
        g5.add_all(k['mpiint/vect_fpts_unpack'])
        g5.add_all(k['mpiint/comm_flux'], deps=k['mpiint/vect_fpts_unpack'])
        g5.add_all(k['eles/tdivtconf'], deps=k['mpiint/comm_flux'])
        for l in k['eles/negdivconf']:
            g5.add(l, deps=edeps(l, 'tdivtconf'))
        g5.commit()

        return [
            (g1, None),
            (g2, m['scal_fpts_send_recv']),
            (g3, m['artvisc_fpts_send_recv']),
            (g4, m['vect_fpts_send_recv']),
            (g5, None)
        ]

    @memoize
    def _compute_grads_graph(self, uinbank):
        m = self._mpireqs
        k, _ = self._get_kernels(uinbank, None)

        def edeps(ek, *edeps): return self._ele_deps(k, ek, *edeps)

        g1 = self.backend.graph()
        g1.add_all(k['eles/disu'])
        g1.add_all(k['mpiint/scal_fpts_pack'], deps=k['eles/disu'])
        g1.commit()

        g2 = self.backend.graph()
        g2.add_all(k['iint/copy_fpts'])
        g2.add_all(k['iint/con_u'], deps=k['iint/copy_fpts'])
        g2.add_all(k['bcint/con_u'], deps=k['iint/copy_fpts'])
        g2.add_all(k['eles/tgradpcoru_upts'])
        g2.commit()

        g3 = self.backend.graph()
        g3.add_all(k['mpiint/scal_fpts_unpack'])
        g3.add_all(k['mpiint/con_u'], deps=k['mpiint/scal_fpts_unpack'])
        g3.add_all(k['eles/tgradcoru_upts'], deps=k['mpiint/con_u'])
        for l in k['eles/gradcoru_upts_curved']:
            g3.add(l, deps=edeps(l, 'tgradcoru_upts'))
        for l in k['eles/gradcoru_upts_linear']:
            g3.add(l, deps=edeps(l, 'tgradcoru_upts'))
        g3.commit()

        return [(g1, None), (g2, m['scal_fpts_send_recv']), (g3, None)]
