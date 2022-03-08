# -*- coding: utf-8 -*-

from collections import defaultdict
from configparser import NoOptionError

from pyfr.integrators.base import BaseCommon


class BaseDualPseudoIntegrator(BaseCommon):
    formulation = 'dual'
    aux_nregs = 0

    def __init__(self, backend, systemcls, rallocs, mesh,
                 initsoln, cfg, stepper_nregs, stage_nregs, dt):
        self.backend = backend
        self.rallocs = rallocs
        self.isrestart = initsoln is not None
        self.cfg = cfg
        self._dt = dt

        sect = 'solver-time-integrator'

        self._dtaumin = 1.0e-12
        self._dtau = cfg.getfloat(sect, 'pseudo-dt')

        self.maxniters = cfg.getint(sect, 'pseudo-niters-max', 0)
        self.minniters = cfg.getint(sect, 'pseudo-niters-min', 0)

        if self.maxniters < self.minniters:
            raise ValueError('The maximum number of pseudo-iterations must '
                             'be greater than or equal to the minimum')

        if (self.pseudo_controller_needs_lerrest and
            not self.pseudo_stepper_has_lerrest):
            raise TypeError('Incompatible pseudo-stepper/pseudo-controller '
                            'combination')

        # Amount of stage storage required by DIRK stepper
        self.stage_nregs = stage_nregs

        # Amount of temp storage required by physical stepper
        self.stepper_nregs = stepper_nregs

        source_nregs = 1

        # Determine the amount of temp storage required in total
        self.nregs = (self.pseudo_stepper_nregs + self.stepper_nregs +
                      self.stage_nregs + source_nregs + self.aux_nregs)

        # Construct the relevant system
        self.system = systemcls(backend, rallocs, mesh, initsoln,
                                nregs=self.nregs, cfg=cfg)

        # Register index list and current index
        self._regidx = list(range(self.nregs))
        self._idxcurr = 0

        # Get a queue for the pseudointegrator
        self._queue = backend.queue()

        # Global degree of freedom count
        self._gndofs = self._get_gndofs()

        elementscls = self.system.elementscls
        self._subdims = [elementscls.convarmap[self.system.ndims].index(v)
                         for v in elementscls.dualcoeffs[self.system.ndims]]

        # Convergence tolerances
        self._pseudo_residtol = residtol = []
        for v in elementscls.convarmap[self.system.ndims]:
            try:
                residtol.append(cfg.getfloat(sect, 'pseudo-resid-tol-' + v))
            except NoOptionError:
                residtol.append(cfg.getfloat(sect, 'pseudo-resid-tol'))

        self._pseudo_norm = cfg.get(sect, 'pseudo-resid-norm', 'l2')
        if self._pseudo_norm not in {'l2', 'uniform'}:
            raise ValueError('Invalid pseudo-residual norm')

        # Pointwise kernels for the pseudo-integrator
        self.pintgkernels = defaultdict(list)

        # Pseudo-step counter
        self.npseudosteps = 0

    @property
    def _pseudo_stepper_regidx(self):
        return self._regidx[:self.pseudo_stepper_nregs]

    @property
    def _source_regidx(self):
        sr = self.pseudo_stepper_nregs + self.stepper_nregs + self.stage_nregs
        return self._regidx[sr]

    @property
    def _stage_regidx(self):
        bsnregs = self.pseudo_stepper_nregs + self.stepper_nregs
        return self._regidx[bsnregs:bsnregs + self.stage_nregs]

    @property
    def _stepper_regidx(self):
        psnregs = self.pseudo_stepper_nregs
        return self._regidx[psnregs:psnregs + self.stepper_nregs]

    def init_stage(self, currstg, stepper_coeffs, dt):
        self.currstg = currstg
        self.stepper_coeffs = stepper_coeffs
        self._dt = dt

        svals = [0, 1/self._dt] + self.stepper_coeffs[:-1]
        sregs = ([self._source_regidx] + self._stepper_regidx
                 + self._stage_regidx[:self.currstg])

        # Accumulate physical stepper sources into a single register
        self._addv(svals, sregs, subdims=self._subdims)

    def store_current_soln(self):
        # Copy the current soln into the first source register
        self._add(0, self._stepper_regidx[0], 1, self._idxcurr)

    def obtain_solution(self, bcoeffs):
        consts = [0, 1] + bcoeffs
        regidxs = ([self._idxcurr, self._stepper_regidx[0]]
                   + self._stage_regidx)

        self._addv(consts, regidxs, subdims=self._subdims)
