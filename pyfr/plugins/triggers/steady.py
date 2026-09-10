import numpy as np

from pyfr.plugins.triggers.base import HistoryTriggerSource
from pyfr.series import mean_ci, mser


class SteadyTriggerSource(HistoryTriggerSource):
    name = 'steady'

    def __init__(self, cfg, cfgsect, manager, intg):
        super().__init__(cfg, cfgsect, manager, intg)

        self._criterion = cfg.get(cfgsect, 'criterion', 'range')

        if self._criterion == 'mser':
            # Transient detection needs the full history, not a window
            self._margin = cfg.getfloat(cfgsect, 'margin')
            self._hist = []
            self._nseen = 0
            self._last = False
        elif self._criterion in {'range', 'gradient', 'std'}:
            self._window = cfg.getint(cfgsect, 'window')
            self._tol = cfg.getfloat(cfgsect, 'tolerance')
        else:
            raise ValueError('Invalid criterion')

    def evaluate(self, intg):
        if self._criterion == 'mser':
            return self._evaluate_mser(intg)
        else:
            return self._evaluate_window(intg)

    def _evaluate_mser(self, intg):
        s = self._series
        if s.n < self._min_samples:
            return False

        # Recompute only when new samples have been published
        if s.n != self._nseen:
            self._nseen = s.n

            t, v = s.t, s.x
            self._hist.append((intg.tcurr, t[mser(t, v)]))

            # Retain only estimates from the stability window
            tmin = intg.tcurr - self._margin
            self._hist = [h for h in self._hist if h[0] >= tmin]

            ds = np.array([d for _, d in self._hist])

            # The estimate must be old, well-covered, and stable
            aged = intg.tcurr - ds[-1] >= self._margin
            covered = self._hist[0][0] <= tmin + 0.1*self._margin
            stable = np.ptp(ds) <= 0.25*self._margin

            self._last = aged and covered and stable

        return self._last

    def _evaluate_window(self, intg):
        s = self._series
        if s.n < self._window:
            return False

        t, v = s.t[-self._window:], s.x[-self._window:]

        mean = np.mean(v)
        if abs(mean) < 1e-30:
            return False

        match self._criterion:
            case 'range':
                metric = np.ptp(v) / abs(mean)
            case 'gradient':
                if np.ptp(t) < 1e-30:
                    return False
                metric = abs(np.polyfit(t, v, 1)[0] / mean)
            case 'std':
                metric = np.std(v) / abs(mean)

        return metric < self._tol


class MeanCITriggerSource(HistoryTriggerSource):
    name = 'meanci'
    has_checkpoint = True

    def __init__(self, cfg, cfgsect, manager, intg):
        super().__init__(cfg, cfgsect, manager, intg)

        self._tol = cfg.getfloat(cfgsect, 'tolerance')
        self._rel = cfg.getbool(cfgsect, 'relative', False)
        self._level = cfg.getfloat(cfgsect, 'level', 0.68)

        # Intervals are meaningless without enough decorrelated samples
        self._minneff = cfg.getfloat(cfgsect, 'min-neff', 32)

        self.after = cfg.get(cfgsect, 'after', None)

        self._start_t = None
        self._nseen = 0
        self._last = False

    def trigger_refs(self):
        return (self.after,) if self.after else ()

    def evaluate(self, intg):
        # Consider only samples after the predecessor became active
        if self.after is not None:
            if not self.manager.active(self.after):
                return False

            if self._start_t is None:
                self._start_t = intg.tcurr

        t, v = self._series.t, self._series.x
        if self._start_t is not None:
            mask = t >= self._start_t
            t, v = t[mask], v[mask]

        if len(t) < self._min_samples:
            return False

        # Recompute only when new samples have been published
        if len(t) != self._nseen:
            self._nseen = len(t)

            m, se, hw, tau, neff = mean_ci(t, v, self._level)

            tol = self._tol*abs(m) if self._rel else self._tol
            self._last = neff >= self._minneff and hw <= tol

        return self._last

    def checkpoint(self):
        st = self._start_t if self._start_t is not None else np.nan
        return np.void((st,), dtype=[('start_t', 'f8')])

    def restore_checkpoint(self, data):
        if not np.isnan(st := data['start_t']):
            self._start_t = float(st)
