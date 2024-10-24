from collections import defaultdict
import itertools as it
import re

import numpy as np

from pyfr.cache import memoize
from pyfr.inifile import Inifile
from pyfr.integrators.dual.pseudo.base import BaseDualPseudoIntegrator
from pyfr.integrators.dual.pseudo.pseudocontrollers import (
    BaseDualPseudoController
)
from pyfr.util import subclass_where


class DualMultiPIntegrator(BaseDualPseudoIntegrator):
    def __init__(self, backend, systemcls, mesh, initsoln, cfg, stepper_nregs,
                 stage_nregs, dt):
        self.backend = backend

        sect = 'solver-time-integrator'
        mgsect = 'solver-dual-time-integrator-multip'

        # Get the solver order and set the initial multigrid level
        self._order = self.level = order = cfg.getint('solver', 'order')

        # Get the multigrid cycle
        self.cycle, self.csteps = zip(*cfg.getliteral(mgsect, 'cycle'))
        self._fgen = np.random.Generator(np.random.PCG64(0))

        self.levels = sorted(set(self.cycle), reverse=True)

        if max(self.cycle) > self._order:
            raise ValueError('The multigrid level orders cannot exceed '
                             'the solution order')

        if any(abs(i - j) > 1 for i, j in zip(self.cycle, self.cycle[1:])):
            raise ValueError('The orders of consecutive multigrid levels can '
                             'only change by one')

        if self.cycle[0] != self._order or self.cycle[-1] != self._order:
            raise ValueError('The multigrid cycle needs to start end with the '
                             'highest (solution) order ')

        # Initialise the number of cycles
        self.npmgcycles = 0

        # Multigrid pseudo-time steps
        dtau = cfg.getfloat(sect, 'pseudo-dt')
        self.dtauf = cfg.getfloat(mgsect, 'pseudo-dt-fact', 1.0)

        self._maxniters = cfg.getint(sect, 'pseudo-niters-max', 0)
        self._minniters = cfg.getint(sect, 'pseudo-niters-min', 0)

        # Get the multigrid pseudostepper and pseudocontroller classes
        pn = cfg.get(sect, 'pseudo-scheme')
        cn = cfg.get(sect, 'pseudo-controller')

        cc = subclass_where(BaseDualPseudoController,
                            pseudo_controller_name=cn)
        cc_none = subclass_where(BaseDualPseudoController,
                                 pseudo_controller_name='none')

        # Construct a pseudo-integrator for each level
        from pyfr.integrators.dual.pseudo import get_pseudo_stepper_cls

        self.pintgs = {}
        for l in self.levels:
            pc = get_pseudo_stepper_cls(pn, l)

            if l == order:
                bases = [cc, pc]
                mcfg = cfg
            else:
                bases = [cc_none, pc]

                mcfg = Inifile(cfg.tostr())
                mcfg.set('solver', 'order', l)
                mcfg.set(sect, 'pseudo-dt', dtau*self.dtauf**(order - l))

                for s in cfg.sections():
                    if (m := re.match(f'solver-(.*)-mg-p{l}$', s)):
                        mcfg.rename_section(s, f'solver-{m[1]}')

            # A class that bypasses pseudo-controller methods within a cycle
            class lpsint(*bases):
                name = f'MultiPPseudoIntegrator{l}'
                aux_nregs = 2 if l != self._order else 0

                @property
                def _aux_regidx(self):
                    if self.aux_nregs != 0:
                        return self._regidx[-2:]

                @property
                def ntotiters(iself):
                    return self.npmgcycles

                def convmon(self, *args, **kwargs):
                    pass

                def _rhs_with_dts(self, t, uin, fout, mg_add=True):
                    super()._rhs_with_dts(t, uin, fout)

                    # Multigrid r addition
                    if mg_add and self._aux_regidx:
                        self._add(1, fout, -1, self._aux_regidx[0])

            stp_nregs = stepper_nregs if l == self._order else 0
            stg_nregs = stage_nregs if l == self._order else 0

            self.pintgs[l] = lpsint(
                backend, systemcls, mesh, initsoln, mcfg, stp_nregs, stg_nregs,
                dt
            )

        # Get the highest p system from plugins
        self.system = self.pintgs[self._order].system

        # Get the convergence monitoring method
        self.mg_convmon = cc.convmon

        # Initialise the restriction and prolongation matrices
        self._init_proj_mats()

    def commit(self):
        for s in self.pintgs.values():
            s.system.commit()

    @property
    def _idxcurr(self):
        return self.pintg._idxcurr

    @_idxcurr.setter
    def _idxcurr(self, y):
        self.pintg._idxcurr = y

    @property
    def pseudostepinfo(self):
        return self.pintg.pseudostepinfo

    @pseudostepinfo.setter
    def pseudostepinfo(self, y):
        self.pintg.pseudostepinfo = y

    @property
    def _regidx(self):
        return self.pintg._regidx

    @property
    def stage_nregs(self):
        return self.pintg.stage_nregs

    @property
    def stepper_nregs(self):
        return self.pintg.stepper_nregs

    @property
    def pseudo_stepper_nregs(self):
        return self.pintg.pseudo_stepper_nregs

    @property
    def _subdims(self):
        return self.pintg._subdims

    @property
    def pintg(self):
        return self.pintgs[self.level]

    def _init_proj_mats(self):
        self.projmats = defaultdict(list)
        cmat = lambda m: self.backend.const_matrix(m, tags={'align'})

        for l in self.levels[1:]:
            for etype in self.pintg.system.ele_types:
                b1 = self.pintgs[l].system.ele_map[etype].basis.ubasis
                b2 = self.pintgs[l + 1].system.ele_map[etype].basis.ubasis

                self.projmats[l, l + 1].append(cmat(b1.proj_to(b2)))
                self.projmats[l + 1, l].append(cmat(b2.proj_to(b1)))

    @memoize
    def mgproject(self, l1, l1reg, l2, l2reg):
        projk = []
        for i, a in enumerate(self.projmats[l1, l2]):
            b = self.pintgs[l1].system.ele_banks[i][l1reg]
            c = self.pintgs[l2].system.ele_banks[i][l2reg]
            projk.append(self.backend.kernel('mul', a, b, out=c))

        return projk

    @memoize
    def dtauproject(self, l1, l2):
        projk = []
        for i, a in enumerate(self.projmats[l1, l2]):
            b = self.pintgs[l1].dtau_upts[i]
            c = self.pintgs[l2].dtau_upts[i]
            projk.append(self.backend.kernel('mul', a, b, out=c,
                                             alpha=self.dtauf))

        return projk

    def restrict(self, l1, l2):
        l1idxcurr = self.pintgs[l1]._idxcurr
        l2idxcurr = self.pintgs[l2]._idxcurr

        # Prevsoln is used as temporal storage at l1
        rtemp = 0 if l1idxcurr == 1 else 1

        # Restrict the physical source term
        l1src = self.pintgs[l1]._source_regidx
        l2dst = self.pintgs[l2]._source_regidx

        # If at top level evaluate src macros
        if l1 == self._order and self.system.has_src_macros:
            self._add(0, rtemp, 1, l1idxcurr)
            self.system.evalsrcmacros(rtemp)
            self._add(1, rtemp, 1, l1src)
            l1src = rtemp

        self.backend.run_kernels(self.mgproject(l1, l1src, l2, l2dst))

        # Project local dtau field to lower multigrid levels
        if self.pintgs[self._order].pseudo_controller_needs_lerrest:
            self.backend.run_kernels(self.dtauproject(l1, l2))

        # rtemp = R = -∇·f - dQ/dt
        self.pintg._rhs_with_dts(self.tcurr, l1idxcurr, rtemp, mg_add=False)

        # rtemp = -d = R - r at lower levels
        if l1 != self._order:
            self.pintg._add(1, rtemp, -1, self._mg_regidx[0])

        # Activate l2 system and get l2 regidx
        self.level = l2
        mg0, mg1 = self._mg_regidx

        # Restrict Q and d
        self.backend.run_kernels(
            self.mgproject(l1, l1idxcurr, l2, l2idxcurr) +
            self.mgproject(l1, rtemp, l2, mg1)
        )

        # mg0 = R = -∇·f - dQ/dt
        self.pintg._rhs_with_dts(self.tcurr, l2idxcurr, mg0, mg_add=False)

        # Compute the target residual r
        # mg0 = r = R + d
        self.pintg._add(1, mg0, -1, mg1)

        # Need to store the non-smoothed solution Q^ns for the correction
        # mg1 = Q^ns
        self.pintg._add(0, mg1, 1, l2idxcurr)

    def prolongate(self, l1, l2):
        l1idxcurr = self.pintgs[l1]._idxcurr
        l2idxcurr = self.pintgs[l2]._idxcurr

        # Prevsoln is used as temporal storage at l2
        rtemp = 0 if l2idxcurr == 1 else 1

        # Correction with respect to the non-smoothed value from down-cycle
        # mg1 = Delta = Q^s - Q^ns
        self.pintg._add(-1, self._mg_regidx[1], 1, l1idxcurr)

        # Prolongate the correction and store to rtemp
        self.backend.run_kernels(
            self.mgproject(l1, self._mg_regidx[1], l2, rtemp)
        )

        # Add the correction to the end quantity at l2
        # Q^m+1  = Q^s + Delta
        self.level = l2
        self.pintg._add(1, l2idxcurr, 1, rtemp)

    @property
    def _mg_regidx(self):
        if self.level == self._order:
            raise AttributeError('_mg_regidx not defined when'
                                 ' self.level == self._order')

        return self.pintg._aux_regidx

    def pseudo_advance(self, tcurr):
        # Multigrid levels and step counts
        cycle, cstepsf = self.cycle, self.csteps

        # Set time step and current stepper coefficients for all levels
        for l in self.levels:
            self.pintgs[l]._dt = self._dt
            self.pintgs[l].stepper_coeffs = self.stepper_coeffs

        self.tcurr = tcurr

        for i in range(self._maxniters):
            # Choose either ⌊c⌋ or ⌈c⌉ in a way that the average is c
            csteps = [int(c + (self._fgen.random() < c % 1)) for c in cstepsf]

            for l, m, n in it.zip_longest(cycle, cycle[1:], csteps):
                self.level = l

                # Set the number of smoothing steps at each level
                self.pintg.maxniters = self.pintg.minniters = n

                self.pintg.pseudo_advance(tcurr)

                if m is not None and l > m:
                    self.restrict(l, m)
                elif m is not None and l < m:
                    self.prolongate(l, m)

            # Update the number of p-multigrid cycles
            self.npmgcycles += 1

            # Convergence monitoring
            if self.mg_convmon(self.pintg, i, self._minniters):
                break

    def collect_stats(self, stats):
        # Collect the stats for each level
        for l in self.levels:
            # Total number of RHS evaluations
            stats.set('solver-time-integrator', f'nfevals-p{l}',
                      self.pintgs[l].pseudo_stepper_nfevals)

            # Total number of pseudo-steps
            stats.set('solver-time-integrator', f'npseudosteps-p{l}',
                      self.pintgs[l].npseudosteps)

        # Total number of p-multigrid cycles
        stats.set('solver-time-integrator', 'npmgcycles', self.npmgcycles)
