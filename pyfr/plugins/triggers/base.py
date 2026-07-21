import re
import time
from collections import defaultdict

import numpy as np

from pyfr.inifile import process_expr
from pyfr.mpiutil import get_comm_rank_root


_cmp_ops = {
    '<': lambda a, b: a < b,
    '<=': lambda a, b: a <= b,
    '>': lambda a, b: a > b,
    '>=': lambda a, b: a >= b,
}

class _Series:
    dtype = np.dtype([('t', 'f8'), ('x', 'f8')])

    def __init__(self, cap=256):
        self._buf = np.empty(cap, dtype=self.dtype)
        self.n = 0

    def push(self, t, x):
        # Grow the backing store when full, giving amortised O(1) pushes
        if self.n == len(self._buf):
            self._buf = np.resize(self._buf, 2*self.n)

        self._buf[self.n] = (t, x)
        self.n += 1

    @property
    def t(self):
        return self._buf['t'][:self.n]

    @property
    def x(self):
        return self._buf['x'][:self.n]

    def dump(self):
        return self._buf[:self.n].copy()

    def load(self, arr):
        self._buf = np.asarray(arr, dtype=self.dtype).copy()
        self.n = len(self._buf)


class BaseTriggerSource:
    name = None
    collective = False
    has_checkpoint = False
    has_history = False

    def __init__(self, cfg, cfgsect, manager, intg):
        self.cfg = cfg
        self.cfgsect = cfgsect
        self.manager = manager

        self.mode = cfg.get(cfgsect, 'mode', 'latch')
        if self.mode not in {'latch', 'level', 'edge'}:
            raise ValueError('Invalid trigger mode')

    def checkpoint(self):
        return None

    def restore_checkpoint(self, data):
        pass

    def _parse_condition(self, cfg, cfgsect, valid_reds):
        c = cfg.items_as('constants', float)

        m = re.match(r'(min|max|sum|avg|l2norm)\((.+)\)\s*(<=?|>=?)\s*(.+)$',
                     cfg.get(cfgsect, 'condition'))
        if not m:
            raise ValueError('Invalid condition syntax')

        red = m[1]
        expr = process_expr(m[2], c)
        cmp = _cmp_ops[m[3]]
        tstr = m[4].strip()
        threshold = float(c.get(tstr, tstr))

        if red not in valid_reds:
            raise ValueError('Invalid reduction')

        return red, expr, cmp, threshold

    def trigger_refs(self):
        return ()

    def evaluate(self, intg):
        raise NotImplementedError


class HistoryTriggerSource(BaseTriggerSource):
    has_history = True

    # Minimum samples before a statistical estimate is attempted
    _min_samples = 16

    def __init__(self, cfg, cfgsect, manager, intg):
        super().__init__(cfg, cfgsect, manager, intg)

        self._source = cfg.get(cfgsect, 'source')
        self._series = manager.subscribe(self._source)

    def dump_history(self):
        return self._series.dump()

    def load_history(self, arr):
        self._series.load(arr)


class TriggerManager:
    def __init__(self):
        self._triggers = {}
        self._states = {}
        self._prev_raw = {}
        self._latest = {}
        self._subs = defaultdict(list)
        self._publishers = {}
        self.wtime_start = time.monotonic()

    def parse_config(self, intg):
        from pyfr.plugins.triggers import get_trigger

        for s in intg.cfg.sections():
            if (m := re.match(r'trigger-(.+)$', s)):
                name = m[1]
                ttype = intg.cfg.get(s, 'type')
                src = get_trigger(ttype, intg.cfg, s, self, intg)
                self._triggers[name] = src
                self._states[name] = False
                self._prev_raw[name] = False

        # Validate cross-trigger references
        for src in self._triggers.values():
            self.check_names(src.trigger_refs())

        if self._triggers:
            self._register_serialiser(intg)

    def __bool__(self):
        return bool(self._triggers)

    def __iter__(self):
        return iter(self._triggers)

    def evaluate(self, intg):
        comm, rank, root = get_comm_rank_root()

        # Phase 1: collective triggers (all ranks participate)
        coll_raw = {}
        for name, src in self._triggers.items():
            if src.collective:
                coll_raw[name] = src.evaluate(intg)

        # Phase 2: root evaluates everything else
        if rank == root:
            for name, src in self._triggers.items():
                raw = coll_raw[name] if src.collective else src.evaluate(intg)
                prev_raw = self._prev_raw.get(name, False)

                match src.mode:
                    case 'latch':
                        new = self._states[name] or raw
                    case 'level':
                        new = raw
                    case 'edge':
                        new = raw and not prev_raw

                self._states[name] = new
                self._prev_raw[name] = raw

        # Phase 3: broadcast all trigger states to all ranks
        names = list(self)

        if rank == root:
            sv = [self._states[n] for n in names]
        else:
            sv = None

        sv = comm.bcast(sv, root=root)

        for name, active in zip(names, sv):
            self._states[name] = active

    def _check_name(self, name):
        if name not in self._triggers:
            raise KeyError(f'Unknown trigger: {name!r}')

    def check_names(self, names):
        for name in names:
            self._check_name(name)

    def active(self, name):
        self._check_name(name)
        return self._states[name]

    def fire(self, name):
        self._check_name(name)
        self._states[name] = True
        self._prev_raw[name] = True

    def register_publisher(self, name, cfgsect):
        if name in self._publishers:
            raise ValueError(f'Publish namespace {name!r} used by both '
                             f'{self._publishers[name]} and {cfgsect}')

        self._publishers[name] = cfgsect

    def subscribe(self, source):
        series = _Series()
        self._subs[source].append(series)
        return series

    def publish(self, name, t, values):
        for field, val in values.items():
            key = f'{name}.{field}'
            self._latest[key] = val
            for series in self._subs[key]:
                series.push(t, val)

    def latest_published(self):
        return self._latest

    def _register_serialiser(self, intg):
        _, rank, root = get_comm_rank_root()

        def reg(prefix, fn):
            intg.serialiser.register(prefix, fn if rank == root else None)

        def datafn():
            names, states = zip(*self._states.items())
            prev_raw = [self._prev_raw[n] for n in names]
            nmax = max(len(n) for n in names)

            dt = [('name', f'S{nmax}'), ('state', '?'), ('prev_raw', '?')]
            return np.array(list(zip(names, states, prev_raw)), dtype=dt)

        reg('triggers', datafn)

        # Per-source scalar state (e.g. duration) and consumed history
        for name, src in self._triggers.items():
            if src.has_checkpoint:
                reg(f'trigger-src/{name}', src.checkpoint)
            if src.has_history:
                reg(f'trigger-pub/{name}', src.dump_history)

    def restore(self, state):
        if state is None:
            return

        if (sdata := state.get('triggers')) is not None:
            for row in sdata:
                name = row['name'].decode()
                if name in self._states:
                    self._states[name] = bool(row['state'])
                    self._prev_raw[name] = bool(row['prev_raw'])

        for name, src in self._triggers.items():
            if src.has_checkpoint:
                if (sd := state.get(f'trigger-src/{name}')) is not None:
                    src.restore_checkpoint(sd)
            if src.has_history:
                if (sd := state.get(f'trigger-pub/{name}')) is not None:
                    src.load_history(sd)
