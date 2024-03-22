from argparse import Action
import contextlib
import itertools as it
import shutil
import sys
import time


def format_bytes(n, dps=2):
    labels = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']

    for l in labels:
        if n < 1024 - 0.5*10**-dps:
            break

        if l != labels[-1]:
            n /= 1024

    return f'{n:.{dps}f} {l}' if l != 'B' else f'{n} B'


def format_hms(delta):
    if delta is not None:
        hours, remainder = divmod(int(delta), 3600)
        minutes, seconds = divmod(remainder, 60)

        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    else:
        return '--:--:--'


def format_s(delta):
    if delta is not None:
        return f'{delta:5.2f}s'
    else:
        return ' 0.00s'


class ProgressBar:
    _dispfmt = '{:7.1%} [{}{}>{}] {:.{dps}f}/{:.{dps}f} ela: {} rem: {}'

    # Minimum time in seconds between updates
    _mindelta = 0.1

    def __init__(self, *, prefix='', suffix='\n', dps=2, fmt=format_hms):
        self.prefix = prefix
        self.suffix = suffix
        self.dps = dps
        self.fmt = fmt

        self._ncol = shutil.get_terminal_size().columns - len(prefix)

    def start(self, end, *, start=0, curr=None):
        self.ststrt = start
        self.strtrt = curr or start
        self.stend = end

        self._wstart = time.time()
        self._last_len = 0
        self._last_wallt = 0.0

        self._nbarcol = self._ncol - 24 - 2*len(f'{end:.{self.dps}f}')

        sys.stderr.write(self.prefix)

    def start_with_iter(self, iterable, n=None):
        self.start(len(iterable) if n is None else n)

        for i in iterable:
            yield i
            self()

    def __call__(self, t=None):
        if t is None:
            t = getattr(self, 'stcurr', 0) + 1

        self.stcurr = min(t, self.stend)
        self.stelap = self.stcurr - self.strtrt

        self._render()

        if self.stcurr == self.stend:
            sys.stderr.write(self.suffix)

    @property
    def walltime(self):
        return time.time() - self._wstart

    def _render(self):
        wallt = self.walltime
        delta = wallt - self._last_wallt

        # If we have rendered recently then do not do so again
        if delta < self._mindelta and self.stcurr != self.stend:
            return

        # Starting, current, elapsed, and ending simulation times
        st, cu, el, en = self.ststrt, self.stcurr, self.stelap, self.stend

        # Relative times
        rcu, ren = cu - st, en - st

        # Fraction of the simulation we've completed
        frac = rcu / ren

        # Elapsed and estimated remaining wall time
        wela = self.fmt(wallt)
        wrem = self.fmt(wallt*(en - cu)/el if self.stelap > 0 else None)

        # Decide how many '+', '=' and ' ' to output for the progress bar
        n = self._nbarcol - len(wela) - len(wrem) - 1
        nps = int(n * (rcu - el)/ren)
        neq = int(round(n * el/ren))
        nsp = n - nps - neq

        # Render the progress bar
        s = self._dispfmt.format(frac, '+'*nps, '='*neq, ' '*nsp, cu, en, wela,
                                 wrem, dps=self.dps)

        # Erase any existing bar and write the new bar
        sys.stderr.write(f'\x1b[{self._last_len}D\x1b[0K{s}')
        sys.stderr.flush()

        # Update the last bar length and render time
        self._last_len = len(s)
        self._last_wallt = wallt


class NullProgressBar(ProgressBar):
    def __init__(self):
        pass

    def start(self, end, *, start=0, curr=None):
        pass

    def __call__(self, t=None):
        pass


class ProgressSpinner:
    # Minimum time in seconds between updates
    _mindelta = 0.08

    def __init__(self, n=8):
        # Spinner character sequence
        seq = [f'[{" "*i}●{" "*(n - i - 1)}]' for i in range(n)]
        self._schar_cycle = it.cycle(seq + seq[-2:0:-1])

        self._last_wallt = 0
        self._last_nchar = 0

        self()

    def erase(self):
        if self._last_nchar:
            sys.stderr.write(f'\x1b[{self._last_nchar}D\x1b[0K')

    def __call__(self, v=None):
        wallt = time.time()

        # If we have rendered recently then do not do so again
        if wallt - self._last_wallt < self._mindelta:
            return

        # Get the next spinner character
        c = next(self._schar_cycle)

        # Append any additional progress values
        if v is not None:
            c = f'{c} {v}'

        self.erase()
        sys.stderr.write(c)
        sys.stderr.flush()

        # Update the last render time and output character
        self._last_wallt = wallt
        self._last_nchar = len(c)

    def wrap_file_lines(self, iter, n):
        nb = 0

        for i, v in enumerate(iter):
            yield v

            nb += len(v)

            if i % n == 0:
                self(format_bytes(nb))


class NullProgressSpinner(ProgressSpinner):
    def __init__(self):
        pass

    def __call__(self, v=None):
        pass


class ProgressSequence:
    def __init__(self, *, prefix=''):
        self._prefix = prefix

    def _start_phase(self, phase):
        return f'{self._prefix} • {phase} ', time.time()

    def _finish_phase(self, phase, prefix, tstart):
        dt = time.time() - tstart
        sys.stderr.write(f'\x1b[2K\x1b[G{prefix}({dt:.2f}s)\n')

    @contextlib.contextmanager
    def start(self, phase):
        prefix, tstart = self._start_phase(phase)

        sys.stderr.write(prefix)
        sys.stderr.flush()

        yield None

        self._finish_phase(phase, prefix, tstart)

    @contextlib.contextmanager
    def start_with_sequence(self, phase):
        prefix, tstart = self._start_phase(phase)

        sys.stderr.write(prefix + '\n')
        sys.stderr.flush()

        yield ProgressSequence(prefix=self._prefix + '  ')

    @contextlib.contextmanager
    def start_with_bar(self, phase):
        prefix, tstart = self._start_phase(phase)

        yield ProgressBar(prefix=prefix, suffix='', dps=0, fmt=format_s)

        self._finish_phase(phase, prefix, tstart)

    @contextlib.contextmanager
    def start_with_spinner(self, phase):
        prefix, tstart = self._start_phase(phase)

        sys.stderr.write(prefix)

        yield ProgressSpinner()

        self._finish_phase(phase, prefix, tstart)


class NullProgressSequence(ProgressSequence):
    def __init__(self):
        pass

    def __bool__(self):
        return False

    @contextlib.contextmanager
    def start(self, phase):
        yield None

    @contextlib.contextmanager
    def start_with_sequence(self, phase):
        yield self

    @contextlib.contextmanager
    def start_with_spinner(self, phase):
        yield NullProgressSpinner()

    @contextlib.contextmanager
    def start_with_bar(self, phase):
        yield NullProgressBar()


class ProgressSequenceAction(Action):
    def __init__(self, *, nargs=0, default=NullProgressSequence(), **kwargs):
        super().__init__(nargs=nargs, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, ProgressSequence())
