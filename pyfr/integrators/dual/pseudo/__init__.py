# -*- coding: utf-8 -*-

import re

from pyfr.integrators.dual.pseudo.pseudocontrollers import BaseDualPseudoController
from pyfr.integrators.dual.pseudo.pseudosteppers import BaseDualPseudoStepper
from pyfr.integrators.dual.pseudo.multip import DualMultiPIntegrator
from pyfr.util import subclass_where


def get_pseudo_integrator(backend, systemcls, rallocs, mesh,
                          initsoln, cfg, tcoeffs, dt):
    # A new type of integrator allowing multip convergence acceleration
    if 'solver-dual-time-integrator-multip' in cfg.sections():
        return DualMultiPIntegrator(backend, systemcls, rallocs, mesh,
                                    initsoln, cfg, tcoeffs, dt)
    else:
        cn = cfg.get('solver-time-integrator', 'pseudo-controller')
        pn = cfg.get('solver-time-integrator', 'pseudo-scheme')

        cc = subclass_where(BaseDualPseudoController, pseudo_controller_name=cn)
        pc = subclass_where(BaseDualPseudoStepper, pseudo_stepper_name=pn)

        # Determine the integrator name
        name = '_'.join(['dual', cn, pn, 'pseudointegrator'])
        name = re.sub('(?:^|_|-)([a-z])', lambda m: m.group(1).upper(), name)

        pseudointegrator = type(name, (cc, pc), dict(name=name))

        # Construct and return an instance of this new integrator class
        return pseudointegrator(backend, systemcls, rallocs, mesh,
                                initsoln, cfg, tcoeffs, dt)
