# -*- coding: utf-8 -*-

import re

from pyfr.integrators.controllers import BaseController
from pyfr.integrators.steppers import BaseStepper
from pyfr.integrators.writers import H5Writer
from pyfr.util import subclass_where


def get_integrator(backend, systemcls, rallocs, mesh, initsoln, cfg):
    # Look-up the controller, stepper and writer names
    c = cfg.get('solver-time-integrator', 'controller')
    s = cfg.get('solver-time-integrator', 'scheme')
    w = 'HDF5'

    controller = subclass_where(BaseController, controller_name=c)
    stepper = subclass_where(BaseStepper, stepper_name=s)
    writer = H5Writer

    # Determine the integrator name
    name = '_'.join([c, s, w])
    name = re.sub('(?:^|_|-)([a-z])', lambda m: m.group(1).upper(), name)

    # Composite the classes together to form a new type
    integrator = type(name, (controller, stepper, writer), dict(name=name))

    # Construct and return an instance of this new integrator class
    return integrator(backend, systemcls, rallocs, mesh, initsoln, cfg)
