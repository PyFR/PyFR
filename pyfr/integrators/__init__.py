# -*- coding: utf-8 -*-

import re

from pyfr.integrators.controllers import BaseController
from pyfr.integrators.steppers import BaseStepper
from pyfr.integrators.writers import BaseWriter
from pyfr.util import subclass_map


def get_integrator(backend, rallocs, mesh, initsoln, cfg):
    controller_map = subclass_map(BaseController, 'controller_name')
    stepper_map = subclass_map(BaseStepper, 'stepper_name')
    writer_map = subclass_map(BaseWriter, 'writer_name')

    # Look-up the controller, stepper and writer classes
    c = controller_map[cfg.get('solver-time-integrator', 'controller')]
    s = stepper_map[cfg.get('solver-time-integrator', 'scheme')]
    w = writer_map[cfg.get('soln-output', 'format')]

    # Determine the integrator name
    name = '_'.join([c.controller_name, s.stepper_name, w.writer_name])
    name = re.sub('(?:^|_|-)([a-z])', lambda m: m.group(1).upper(), name)

    # Composite the classes together to form a new type
    integrator = type(name, (c, s, w), dict(name=name))

    # Construct and return an instance of this new integrator class
    return integrator(backend, rallocs, mesh, initsoln, cfg)
