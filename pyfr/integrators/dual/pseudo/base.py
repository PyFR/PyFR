# -*- coding: utf-8 -*-

from collections import defaultdict
from configparser import NoOptionError

from pyfr.integrators.base import BaseCommon
from pyfr.util import proxylist


class BaseDualPseudoIntegrator(BaseCommon):
    formulation = 'dual'
    aux_nregs = 0

    def __init__(self, backend, systemcls, rallocs, mesh,
                 initsoln, cfg, stepper_coeffs, dt):
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

        if (self._pseudo_controller_needs_lerrest and
            not self._pseudo_stepper_has_lerrest):
            raise TypeError('Incompatible pseudo-stepper/pseudo-controller '
                            'combination')

        # Amount of temp storage required by physical stepper
        self._stepper_nregs = len(stepper_coeffs) - 1

        # Determine the amount of temp storage required in total
        self.nregs = (self._pseudo_stepper_nregs + self._stepper_nregs +
                      self.aux_nregs)

        # Physical stepper coefficients
        self._stepper_coeffs = stepper_coeffs

        # Construct the relevant system
        self.system = systemcls(backend, rallocs, mesh, initsoln,
                                nregs=self.nregs, cfg=cfg)

        # Storage for register banks and current index
        self._init_reg_banks()

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
        self.pintgkernels = defaultdict(proxylist)

    @property
    def _pseudo_stepper_regidx(self):
        return self._regidx[:self._pseudo_stepper_nregs]

    @property
    def _stepper_regidx(self):
        psnregs = self._pseudo_stepper_nregs
        return self._regidx[psnregs:psnregs + self._stepper_nregs]

    def finalise_pseudo_advance(self, currsoln):
        psnregs = self._pseudo_stepper_nregs

        # Rotate the source registers to the right by one
        self._regidx[psnregs:psnregs + self._stepper_nregs] = (
            self._stepper_regidx[-1:] + self._stepper_regidx[:-1]
        )

        # Copy the current soln into the first source register
        self._add(0, self._regidx[psnregs], 1, currsoln)
