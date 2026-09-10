from pyfr.plugins.base import BasePlugin
from pyfr.plugins.mixins import InSituMixin


class BaseSolnPlugin(InSituMixin, BasePlugin):
    prefix = 'soln'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        cfg, s = self.cfg, cfgsect
        optval = lambda k: cfg.get(s, k) if cfg.hasopt(s, k) else None

        # Trigger configuration
        self.trigger_action = cfg.get(s, 'trigger-action', 'activate')
        self.trigger_write_name = optval('trigger-write')
        self.trigger_fire_name = optval('trigger-set')

        # Parse trigger: & for AND, | for OR, or a single name
        trig = optval('trigger')
        if trig is None:
            self.trigger = None
            self.trigger_comb = None
        elif '&' in trig:
            self.trigger = names = [t.strip() for t in trig.split('&')]
            self.trigger_comb = lambda tm: all(tm.active(n) for n in names)
        elif '|' in trig:
            self.trigger = names = [t.strip() for t in trig.split('|')]
            self.trigger_comb = lambda tm: any(tm.active(n) for n in names)
        else:
            self.trigger = names = [trig.strip()]
            self.trigger_comb = lambda tm: tm.active(names[0])

        # Step frequency gating
        self.nsteps = int(v) if (v := optval('nsteps')) else None
