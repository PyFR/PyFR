import re
import time

import numpy as np

from pyfr.inifile import process_expr
from pyfr.mpiutil import get_comm_rank_root


_cmp_ops = {
    '<': lambda a, b: a < b,
    '<=': lambda a, b: a <= b,
    '>': lambda a, b: a > b,
    '>=': lambda a, b: a >= b,
}


class BaseTriggerSource:
    name = None
    collective = False
    has_checkpoint = False

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


class TriggerManager:
    def __init__(self):
        self._triggers = {}
        self._states = {}
        self._prev_raw = {}
        self._published = {}
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

    def publish(self, name, t, values):
        buf = self._published.setdefault(name, [])
        buf.append((t, dict(values)))

    def get_published(self, source):
        name, _, field = source.rpartition('.')
        if not name:
            raise ValueError(f'Invalid source: {source}')

        buf = self._published.get(name, [])
        return [(t, v[field]) for t, v in buf if field in v]

    def latest_published(self):
        result = {}
        for pname, buf in self._published.items():
            if buf:
                _, vals = buf[-1]
                for field, val in vals.items():
                    result[f'{pname}.{field}'] = val

        return result

    def _register_serialiser(self, intg):
        _, rank, root = get_comm_rank_root()

        def datafn():
            names, states = zip(*self._states.items())
            prev_raw = [self._prev_raw[n] for n in names]
            nmax = max(len(n) for n in names)

            dt = [('name', f'S{nmax}'), ('state', '?'), ('prev_raw', '?')]
            return np.array(list(zip(names, states, prev_raw)), dtype=dt)

        intg.serialiser.register('triggers', datafn if rank == root else None)

        # Per-source checkpoint state (e.g. duration triggers)
        for name, src in self._triggers.items():
            if src.has_checkpoint:
                intg.serialiser.register(
                    f'trigger-src/{name}',
                    src.checkpoint if rank == root else None
                )

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
