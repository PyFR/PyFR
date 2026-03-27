import re
import time

import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.nputil import npeval
from pyfr.inifile import process_expr
from pyfr.plugins.triggers.base import BaseTriggerSource, _cmp_ops


class ManualTriggerSource(BaseTriggerSource):
    name = 'manual'

    def evaluate(self, intg):
        return False


class TimeTriggerSource(BaseTriggerSource):
    name = 'time'

    def __init__(self, cfg, cfgsect, manager, intg):
        super().__init__(cfg, cfgsect, manager, intg)
        self.t = cfg.getfloat(cfgsect, 't')

    def evaluate(self, intg):
        return intg.tcurr >= self.t


class WallclockTriggerSource(BaseTriggerSource):
    name = 'wallclock'

    def __init__(self, cfg, cfgsect, manager, intg):
        super().__init__(cfg, cfgsect, manager, intg)
        self.t = cfg.getfloat(cfgsect, 't')

    def evaluate(self, intg):
        elapsed = time.monotonic() - self.manager.wtime_start
        return elapsed >= self.t


class SignalTriggerSource(BaseTriggerSource):
    name = 'signal'

    def __init__(self, cfg, cfgsect, manager, intg):
        import signal

        super().__init__(cfg, cfgsect, manager, intg)

        signame = cfg.get(cfgsect, 'signal').upper().removeprefix('SIG')
        signame = 'SIG' + signame

        signum = getattr(signal, signame, None)
        if signum is None:
            raise ValueError(f'Unknown signal: {signame}')

        self._sigset = {signum}

        # Only root masks the signal
        _, rank, root = get_comm_rank_root()
        self._is_root = rank == root
        if self._is_root:
            signal.pthread_sigmask(signal.SIG_BLOCK, self._sigset)

    def evaluate(self, intg):
        import signal

        if self._is_root and self._sigset & signal.sigpending():
            signal.sigwait(self._sigset)
            return True
        else:
            return False


class FileTriggerSource(BaseTriggerSource):
    name = 'file'
    _watch_modes = {'exists', 'mtime', 'ctime', 'atime'}

    def __init__(self, cfg, cfgsect, manager, intg):
        user_set_mode = cfg.hasopt(cfgsect, 'mode')

        super().__init__(cfg, cfgsect, manager, intg)
        self.path = cfg.getpath(cfgsect, 'path', abs=True)

        self._watch = cfg.get(cfgsect, 'watch', 'exists')
        if self._watch not in self._watch_modes:
            raise ValueError('Invalid watch mode')

        # For time-based watching, default to level (not latch) since
        # the raw condition is inherently pulse-like
        if self._watch != 'exists' and not user_set_mode:
            self.mode = 'level'

        # Snapshot the current stat time so we don't fire immediately
        self._last_time = self._stat_time()

    @property
    def has_checkpoint(self):
        return self._watch != 'exists'

    def _stat_time(self):
        try:
            return getattr(self.path.stat(), f'st_{self._watch}_ns', None)
        except OSError:
            return None

    def evaluate(self, intg):
        # Existence-based; fire when the file appears or disappears
        if self._watch == 'exists':
            return self.path.exists()
        # Time-based; fire when the watched time changes
        else:
            if (t := self._stat_time()) is not None and t != self._last_time:
                self._last_time = t
                return True
            else:
                return False

    def checkpoint(self):
        return np.void((self._last_time or -1,), dtype=[('last_time', 'i8')])

    def restore_checkpoint(self, data):
        if (lt := int(data['last_time'])) >= 0:
            self._last_time = lt


class ExpressionTriggerSource(BaseTriggerSource):
    name = 'expression'

    def __init__(self, cfg, cfgsect, manager, intg):
        super().__init__(cfg, cfgsect, manager, intg)

        c = cfg.items_as('constants', float)
        m = re.match(r'(.+)\s*(<=?|>=?)\s*(\S+)\s*$',
                     cfg.get(cfgsect, 'condition'))
        if not m:
            raise ValueError('Invalid condition syntax')

        raw = m[1].strip()
        self._has_pub = bool(re.search(r'[a-zA-Z]\.[a-zA-Z]', raw))
        self._expr = process_expr(raw, c)
        self._cmp = _cmp_ops[m[2]]
        tstr = m[3].strip()
        self._threshold = float(c.get(tstr, tstr))

    def evaluate(self, intg):
        # Substitute published values (name.field) into the expression
        if pub := self.manager.latest_published():
            p = '|'.join(re.escape(k) for k in pub)
            expr = re.sub(rf'(?<![.\w])({p})(?![.\w])',
                          lambda m: str(pub[m[1]]), self._expr)
        elif self._has_pub:
            return False
        else:
            expr = self._expr

        subs = {'t': intg.tcurr, 'dt': intg.dt, 'step': intg.nacptsteps}
        return self._cmp(npeval(expr, subs), self._threshold)


class DurationTriggerSource(BaseTriggerSource):
    name = 'duration'
    has_checkpoint = True

    def __init__(self, cfg, cfgsect, manager, intg):
        super().__init__(cfg, cfgsect, manager, intg)

        self.after = cfg.get(cfgsect, 'after')

        if cfg.hasopt(cfgsect, 'duration'):
            self._dur = cfg.getfloat(cfgsect, 'duration')
            self._use_steps = False
        elif cfg.hasopt(cfgsect, 'steps'):
            self._dur = cfg.getint(cfgsect, 'steps')
            self._use_steps = True
        else:
            raise ValueError('duration or steps required')

        self._start_t = None
        self._start_step = None

    def trigger_refs(self):
        return (self.after,)

    def evaluate(self, intg):
        if not self.manager.active(self.after):
            return False

        # Record when the 'after' trigger first became active
        if self._start_t is None:
            self._start_t = intg.tcurr
            self._start_step = intg.nacptsteps

        if self._use_steps:
            return intg.nacptsteps - self._start_step >= self._dur
        else:
            return intg.tcurr - self._start_t >= self._dur

    def checkpoint(self):
        st = self._start_t if self._start_t is not None else np.nan
        ss = self._start_step if self._start_step is not None else -1
        dt = [('start_t', 'f8'), ('start_step', 'i8')]
        return np.void((st, ss), dtype=dt)

    def restore_checkpoint(self, data):
        if not np.isnan(st := float(data['start_t'])):
            self._start_t = st
            self._start_step = int(data['start_step'])


class _CompositeTriggerSource(BaseTriggerSource):
    def __init__(self, cfg, cfgsect, manager, intg):
        super().__init__(cfg, cfgsect, manager, intg)
        self.triggers = cfg.get(cfgsect, 'triggers').split()

    def trigger_refs(self):
        return self.triggers

    def evaluate(self, intg):
        return self._fn(self.manager.active(n) for n in self.triggers)


class AllTriggerSource(_CompositeTriggerSource):
    name = 'all'
    _fn = staticmethod(all)


class AnyTriggerSource(_CompositeTriggerSource):
    name = 'any'
    _fn = staticmethod(any)
