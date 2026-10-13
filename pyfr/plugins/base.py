import re


class BasePlugin:
    name = None
    systems = None
    dimensions = None
    suffix = None
    enabled = True
    nsteps = None
    trigger_cond = None
    trigger_action = 'activate'
    trigger_activated = False
    trigger_actions = frozenset()
    trigger_binds = {}
    trigger_fire_name = None

    def __init__(self, *, cfg=None, cfgsect=None, ndims=None):
        self.cfg = cfg
        self.cfgsect = cfgsect
        self.ndims = ndims

        if self.systems and cfg is not None:
            system = cfg.get('solver', 'system')
            if not re.fullmatch(self.systems, system):
                raise RuntimeError(f'System {system} not supported by '
                                   f'plugin {self.name}')

        if self.dimensions and ndims is not None:
            if not re.fullmatch(self.dimensions, str(ndims)):
                raise RuntimeError(f'Dimensionality of {ndims} not '
                                   f'supported by plugin {self.name}')

    def __call__(self, intg):
        pass

    def start(self, intg):
        pass

    def trigger_for(self, intg, action):
        raise NotImplementedError

    def finalise(self, intg):
        pass


class BaseCLIPlugin:
    name = None

    @classmethod
    def add_cli(cls, parser):
        pass
