import numpy as np

from pyfr.plugins.triggers.base import BaseTriggerSource


class SteadyTriggerSource(BaseTriggerSource):
    name = 'steady'

    def __init__(self, cfg, cfgsect, manager, intg):
        super().__init__(cfg, cfgsect, manager, intg)

        self._source = cfg.get(cfgsect, 'source')
        self._window = cfg.getint(cfgsect, 'window')
        self._tol = cfg.getfloat(cfgsect, 'tolerance')
        self._criterion = cfg.get(cfgsect, 'criterion', 'range')

        if self._criterion not in {'range', 'gradient', 'std'}:
            raise ValueError('Invalid criterion')

        self._tarr, self._varr = np.empty((2, self._window))
        self._pos = 0

    def evaluate(self, intg):
        data = self.manager.get_published(self._source)
        w = self._window

        # Ingest new samples into ring buffer
        for t, v in data[self._pos:]:
            i = self._pos % w
            self._tarr[i], self._varr[i] = t, v
            self._pos += 1

        if self._pos < self._window:
            return False

        mean = np.mean(self._varr)
        if abs(mean) < 1e-30:
            return False

        match self._criterion:
            case 'range':
                metric = np.ptp(self._varr) / abs(mean)
            case 'gradient':
                if np.ptp(self._tarr) < 1e-30:
                    return False
                slope = np.polyfit(self._tarr, self._varr, 1)[0]
                metric = abs(slope / mean)
            case 'std':
                metric = np.std(self._varr) / abs(mean)

        return metric < self._tol
