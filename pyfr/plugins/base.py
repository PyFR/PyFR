class BasePlugin:
    name = None
    systems = None
    dimensions = None
    suffix = None
    enabled = True
    nsteps = None
    trigger = None
    trigger_comb = None
    trigger_action = 'activate'
    trigger_activated = False
    trigger_write_name = None
    trigger_fire_name = None

    def __init__(self, *, cfg=None, cfgsect=None, ndims=None):
        self.cfg = cfg
        self.cfgsect = cfgsect
        self.ndims = ndims

    def __call__(self, intg):
        pass

    def trigger_write(self, intg):
        pass

    def finalise(self, intg):
        pass

    def setup(self, sdata, serialiser):
        pass


class BaseCLIPlugin:
    name = None

    @classmethod
    def add_cli(cls, parser):
        pass
