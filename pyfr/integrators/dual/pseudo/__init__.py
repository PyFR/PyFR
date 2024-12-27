import re

from pyfr.integrators.dual.pseudo.pseudocontrollers import (
    BaseDualPseudoController
)
from pyfr.integrators.dual.pseudo.pseudosteppers import BaseDualPseudoStepper
from pyfr.integrators.dual.pseudo.multip import DualMultiPIntegrator
from pyfr.util import subclass_where


def get_pseudo_stepper_cls(name, porder):
    for p in range(porder, -1, -1):
        try:
            return subclass_where(BaseDualPseudoStepper,
                                  pseudo_stepper_name=name,
                                  pseudo_stepper_porder=p)
        except KeyError:
            pass

    return subclass_where(BaseDualPseudoStepper, pseudo_stepper_name=name)


def get_pseudo_integrator(backend, systemcls, mesh, initsoln, cfg, stepnregs,
                          stagenregs, dt):
    # A new type of integrator allowing multip convergence acceleration
    if 'solver-dual-time-integrator-multip' in cfg.sections():
        return DualMultiPIntegrator(backend, systemcls, mesh, initsoln, cfg,
                                    stepnregs, stagenregs, dt)
    else:
        cn = cfg.get('solver-time-integrator', 'pseudo-controller')
        pn = cfg.get('solver-time-integrator', 'pseudo-scheme')
        porder = cfg.getint('solver', 'order')

        cc = subclass_where(BaseDualPseudoController,
                            pseudo_controller_name=cn)
        pc = get_pseudo_stepper_cls(pn, porder)

        # Determine the integrator name
        name = '_'.join(['dual', cn, pn, 'pseudointegrator'])
        name = re.sub('(?:^|_|-)([a-z])', lambda m: m[1].upper(), name)

        pseudointegrator = type(name, (cc, pc), dict(name=name))

        # Construct and return an instance of this new integrator class
        return pseudointegrator(backend, systemcls, mesh, initsoln, cfg,
                                stepnregs, stagenregs, dt)
