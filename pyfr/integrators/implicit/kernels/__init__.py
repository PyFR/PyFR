import re

from pyfr.integrators.explicit import BaseExplicitController, BaseExplicitStepper
from pyfr.util import subclass_where


def get_integrator(backend, systemcls, mesh, initsoln, cfg):
    cn = cfg.get('solver-time-integrator', 'controller')
    sn = cfg.get('solver-time-integrator', 'scheme')

    cc = subclass_where(BaseExplicitController, controller_name=cn)
    sc = subclass_where(BaseExplicitStepper, stepper_name=sn)

    # Determine the integrator name
    name = '_'.join([cn, sn, 'integrator'])
    name = re.sub('(?:^|_|-)([a-z])', lambda m: m[1].upper(), name)

    # Composite the base classes together to form a new type
    integrator = type(name, (cc, sc), dict(name=name))

    # Construct and return an instance of this new integrator class
    return integrator(backend, systemcls, mesh, initsoln, cfg)
