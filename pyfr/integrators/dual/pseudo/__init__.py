# -*- coding: utf-8 -*-

from pkg_resources import resource_listdir
import re

from pyfr.integrators.dual.pseudo.pseudocontrollers import (
    BaseDualPseudoController
)
from pyfr.integrators.dual.pseudo.pseudosteppers import (
    BaseDualPseudoStepper, DualDenseRKPseudoStepper
)
from pyfr.integrators.dual.pseudo.multip import DualMultiPIntegrator
from pyfr.util import subclass_where


def register_tabulated_pseudo_steppers():
    if hasattr(register_tabulated_pseudo_steppers, '_schemes'):
        return

    register_tabulated_pseudo_steppers._schemes = schemes = []

    # Create subclasses for all tabulated dense schemes
    for path in resource_listdir('pyfr.integrators', 'schemes'):
        m = re.match(r'([a-zA-Z0-9\-~+]+)-s(\d+)-p(\d+)'
                     r'(?:-sp(\d+))?(?:-([e]+))?\.txt', path)
        name = m[1]

        attrs = {'pseudo_stepper_name': name, 'path': path}
        attrs['pseudo_stepper_nregs'] = int(m[2])
        attrs['pseudo_stepper_order'] = int(m[3])
        attrs['pseudo_stepper_has_lerrest'] = bool(m[5])

        if m[4]:
            attrs['pseudo_stepper_porder'] = int(m[4])
            name += m[4]

        schemes.append(type(name, (DualDenseRKPseudoStepper,), attrs))


def get_pseudo_stepper_cls(name, porder):
    for p in range(porder, -1, -1):
        try:
            return subclass_where(BaseDualPseudoStepper,
                                  pseudo_stepper_name=name,
                                  pseudo_stepper_porder=p)
        except KeyError:
            pass

    return subclass_where(BaseDualPseudoStepper, pseudo_stepper_name=name)


def get_pseudo_integrator(backend, systemcls, rallocs, mesh,
                          initsoln, cfg, stepnregs, stagenregs, dt):
    register_tabulated_pseudo_steppers()

    # A new type of integrator allowing multip convergence acceleration
    if 'solver-dual-time-integrator-multip' in cfg.sections():
        return DualMultiPIntegrator(backend, systemcls, rallocs, mesh,
                                    initsoln, cfg, stepnregs, stagenregs, dt)
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
        return pseudointegrator(backend, systemcls, rallocs, mesh,
                                initsoln, cfg, stepnregs, stagenregs, dt)
