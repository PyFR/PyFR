from pyfr.plugins.base import BasePlugin
from pyfr.plugins.mixins import InSituMixin


class BaseSolnPlugin(InSituMixin, BasePlugin):
    prefix = 'soln'
    reserved_trigger_keys = {'trigger-action', 'trigger-set'}

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        cfg, s, tm = self.cfg, cfgsect, intg.triggers
        optval = lambda k: cfg.get(s, k) if cfg.hasopt(s, k) else None

        # Trigger configuration
        self.trigger_action = cfg.get(s, 'trigger-action', 'activate')
        self.trigger_fire_name = optval('trigger-set')

        # Validate plugin trigger references
        if self.trigger_fire_name:
            tm.check_names([self.trigger_fire_name])

        # Parse the trigger condition controlling this plugin
        if trig := optval('trigger'):
            self.trigger_cond = tm.condition(trig)

        # Bind each supported action to its trigger condition
        self.trigger_binds = {}
        for k, v in cfg.items(s, prefix='trigger-').items():
            action = k.removeprefix('trigger-')

            if k in self.reserved_trigger_keys:
                continue
            elif action in self.trigger_actions:
                self.trigger_binds[action] = tm.condition(v)
            else:
                raise ValueError(f'Plugin {self.name} does not support {k}')

        # Step frequency gating
        self.nsteps = int(v) if (v := optval('nsteps')) else None
