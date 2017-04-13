# -*- coding: utf-8 -*-

from abc import abstractmethod

from pyfr.integrators.base import BaseIntegrator
from pyfr.util import memoize, proxylist


class BaseDualIntegrator(BaseIntegrator):
    formulation = 'dual'

    def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
        super().__init__(backend, systemcls, rallocs, mesh, initsoln, cfg)
        sect = 'solver-time-integrator'

        self._dtau = self.cfg.getfloat(sect, 'pseudo-dt')
        self.dtaumin = 1.0e-12

        # Dual Integrator treats following variables as lists to accommodate MG
        self.dualsystems = []
        self._dualregs = []

        self.levels = 1
        nreg = self._stepper_nregs

        # Check whether to employ multigrid
        self.multigrid = self.cfg.get(sect, 'controller') == 'multigrid'

        if self.multigrid:
            # Multigrid requires 3 additional registers
            nreg += 3
            # Number of iterations at each stage
            self._leveliters = list(self.cfg.getliteral(sect, 'mgcycle'))
            # Number of levels
            self.levels = len(self._leveliters)

        for level in range(self.levels):
            self.dualsystems.append(systemcls(backend, rallocs, mesh,
                                              initsoln, nreg, cfg, level))
            self._dualregs.append(self._get_reg_banks(nreg, level=level)[0])

        if self.multigrid:
            self.prolongmat, self.restrictmat = self._get_prolrest_matrices()

        # Indices of current and previous soln tracked between multigrid levels
        self._dualidxcurr = [0]*self.levels
        self._dualidxprev = [0]*self.levels

        # Variables refers to the highest polynomial level for plugin support
        self._dualregidx = self._get_reg_banks(nreg)[1]
        self.system = self.dualsystems[0]
        self._idxcurr = self._dualidxcurr[0]

        # Event handlers for advance_to
        self.completed_step_handlers = proxylist(self._get_plugins())

        # Delete the memory-intensive elements map from the system
        del self.system.ele_map

    def _get_reg_banks(self, nreg, level=0):
        regs, regidx = [], list(range(nreg))

        # Create a proxylist of matrix-banks for each storage register
        for i in regidx:
            regs.append(
                proxylist([self.backend.matrix_bank(em, i)
                           for em in self.dualsystems[level].ele_banks])
            )

        return regs, regidx

    def _prepare_reg_banks(self, *bidxes, level=0):
        for reg, ix in zip(self._dualregs[level], bidxes):
            reg.active = ix

    def _get_kernels(self, name, nargs, level=0, **kwargs):
        # Transpose from [nregs][neletypes] to [neletypes][nregs]
        transregs = zip(*self._dualregs[level])

        # Generate an kernel for each element type
        kerns = proxylist([])
        for tr in transregs:
            kerns.append(self.backend.kernel(name, *tr[:nargs], **kwargs))

        return kerns

    @property
    def _stepper_regidx(self):
        return self._dualregidx[:self._pseudo_stepper_nregs]

    @property
    def _source_regidx(self):
        pnreg, dsrc = self._pseudo_stepper_nregs, self._dual_time_source
        return self._dualregidx[pnreg:pnreg + len(dsrc) - 1]

    @property
    def _mg_regidx(self):
        return self._dualregidx[-3:]

    @abstractmethod
    def _dual_time_source(self):
        pass

    @abstractmethod
    def finalise_step(self, currsoln):
        pass

    @memoize
    def prolrest(self, l1, l2):
        if l2 < l1:
            opmats = self.prolongmat[l1]
        else:
            opmats = self.restrictmat[l1]

        prolrestkerns = proxylist([])

        for i in range(len(self.system.ele_types)):
            prolrestkerns.append(
                self.backend.kernel(
                    'mul', opmats[i],
                    self.dualsystems[l1].eles_scal_upts_inb[i],
                    out=self.dualsystems[l2].eles_scal_upts_inb[i])
            )

        return prolrestkerns

    def _get_prolrest_matrices(self):
        rest = [0]*self.levels
        prol = [0]*self.levels

        for level in range(self.levels-1):
            r = proxylist([])
            p = proxylist([])
            for etype in self.system.ele_types:
                b1 = self.dualsystems[level].ele_map[etype].basis
                b2 = self.dualsystems[level + 1].ele_map[etype].basis
                rmat = b1.ubasis.nodal_basis_at(b2.upts)
                pmat = b2.ubasis.nodal_basis_at(b1.upts)
                r.append(self.backend.const_matrix(rmat, tags={'align'}))
                p.append(self.backend.const_matrix(pmat, tags={'align'}))
            rest[level] = r
            prol[level + 1] = p

        return prol, rest

    def restrict(self, l1, l2, dt):
        l1idxcurr, l2idxcurr = self._dualidxcurr[l1], self._dualidxcurr[l2]
        l1sys, l2sys = self.dualsystems[l1], self.dualsystems[l2]
        mg0, mg1, mg2 = self._mg_regidx

        add, rhs = self._add, self._rhs_with_dts

        # mg1 = R = -∇·f - dQ/dt
        rhs(self.tcurr, l1idxcurr, mg1, c=1/dt, level=l1, passmg=True)

        l = 0 if l1 == 0 else 1
        # mg1 = d = r - R
        add(-1, mg1, l, mg0, level=l1)

        # Restrict Q
        l1sys.eles_scal_upts_inb.active = l1idxcurr
        l2sys.eles_scal_upts_inb.active = l2idxcurr
        self._queue % self.prolrest(l1, l2)()

        # Need to store Q^ns before smoothing for the up-cycle correction
        # mg2 = Q^ns
        add(0.0, mg2, 1, l2idxcurr, level=l2)

        # Restrict d and store to m1
        l1sys.eles_scal_upts_inb.active = mg1
        l2sys.eles_scal_upts_inb.active = mg1
        self._queue % self.prolrest(l1, l2)()

        # mg0 = R = -∇·f - dQ/dt
        rhs(0.0, l2idxcurr, mg0, c=1/dt, level=l2, passmg=True)

        # mg0 = r = R + d
        # r needs to be added to each RHS calculation at lower levels
        # Handled by rhs_with_dts in pseudo-stepper class
        add(1, mg0, 1, mg1, level=l2)

        # Source terms at lower polynomial level
        for srcidx in self._source_regidx:
            l1sys.eles_scal_upts_inb.active = srcidx
            l2sys.eles_scal_upts_inb.active = srcidx
            self._queue % self.prolrest(l1, l2)()

    def prolongate(self, l1, l2):
        l1idxcurr, l2idxcurr = self._dualidxcurr[l1], self._dualidxcurr[l2]
        l1sys, l2sys = self.dualsystems[l1], self.dualsystems[l2]
        mg0, mg1, mg2 = self._mg_regidx

        add = self._add

        # Correction with respect to the non-smoothed value from down-cycle
        # mg1 = Delta = Q^s - Q^ns
        add(0, mg1, 1, l1idxcurr, -1, mg2, level=l1)

        # Prolongate the correction and store to mg1
        l1sys.eles_scal_upts_inb.active = mg1
        l2sys.eles_scal_upts_inb.active = mg1
        self._queue % self.prolrest(l1, l2)()

        # Add the correction to the end quantity at l2
        # Q^m+1  = Q^s + Delta
        add(1, l2idxcurr, 1, mg1, level=l2)
