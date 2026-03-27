import re

from pyfr.integrators.explicit import (BaseExplicitController,
                                       BaseExplicitStepper)
from pyfr.integrators.implicit import (BaseImplicitController,
                                       BaseImplicitStepper)
from pyfr.integrators.implicit.krylov import BaseKrylovSolver
from pyfr.util import subclass_where


def get_integrator(backend, systemcls, mesh, initsoln, cfg):
    cn = cfg.get('solver-time-integrator', 'controller')
    sn = cfg.get('solver-time-integrator', 'scheme')

    form = cfg.get('solver-time-integrator', 'formulation', 'explicit')
    if form == 'explicit':
        cc = subclass_where(BaseExplicitController, controller_name=cn)
        sc = subclass_where(BaseExplicitStepper, stepper_name=sn)
        bases = (cc, sc)
    elif form == 'implicit':
        cc = subclass_where(BaseImplicitController, controller_name=cn)
        sc = subclass_where(BaseImplicitStepper, stepper_name=sn)

        kn = cfg.get('solver-time-integrator', 'krylov-solver', 'gmres')
        kc = subclass_where(BaseKrylovSolver, krylov_name=kn)

        bases = (cc, sc, kc)
    else:
        raise ValueError(f'Invalid formulation {form!r}')

    # Determine the integrator name
    name = '_'.join([form, cn, sn, 'integrator'])
    name = re.sub('(?:^|_|-)([a-z])', lambda m: m[1].upper(), name)

    # Composite the base classes together to form a new type
    integrator = type(name, bases, dict(name=name))

    # Construct and return an instance of this new integrator class
    return integrator(backend, systemcls, mesh, initsoln, cfg)
