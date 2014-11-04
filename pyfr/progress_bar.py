# -*- coding: utf-8 -*-

import sys
import time


def get_terminal_size():
    if sys.platform in ('linux2', 'darwin'):
        import fcntl, termios, struct
        try:
            s = struct.unpack('hh', fcntl.ioctl(0, termios.TIOCGWINSZ, '1234'))
            return s
        except IOError:
            pass

    # Default to 24 by 80
    return 24, 80


def to_hms(delta):
    hours, remainder = divmod(int(delta), 3600)
    minutes, seconds = divmod(remainder, 60)

    return hours, minutes, seconds


def format_hms(delta):
    if delta is not None:
        return '{:02d}:{:02d}:{:02d}'.format(*to_hms(delta))
    else:
        return '--:--:--'


class ProgressBar(object):
    _dispfmt = '{:7.1%} [{}{}>{}] {:.{dps}f}/{:.{dps}f} ela: {} rem: {}'

    # Minimum time in seconds between updates
    _mindelta = 0.1

    def __init__(self, start, curr, end, dps=2):
        self.ststrt = start
        self.strtrt = curr
        self.stend = end
        self.dps = dps

        self._wstart = time.time()
        self._last_wallt = 0.0

        self._ncol = get_terminal_size()[1]
        self._nbarcol = self._ncol - 24 - 2*len('{:.{}f}'.format(end, dps))

        self.advance_to(curr)

    def advance_to(self, t):
        self.stcurr = min(t, self.stend)
        self.stelap = self.stcurr - self.strtrt

        self._render()

        if self.stcurr == self.stend:
            sys.stderr.write('\n')

    @property
    def walltime(self):
        return time.time() - self._wstart

    def _render(self):
        wallt = self.walltime
        delta = wallt - self._last_wallt

        # If we have rendered recently then do not do so again
        if delta < self._mindelta and self.stcurr != self.stend:
            return

        # Starting, current, elapsed and ending simulation times
        st, cu, el, en = self.ststrt, self.stcurr, self.stelap, self.stend

        # Relative times
        rcu, ren = cu - st,  en - st

        # Fraction of the simulation we've completed
        frac = float(rcu) / ren

        # Elapsed and estimated remaining wall time
        wela = format_hms(wallt)
        wrem = format_hms(wallt*(en - cu)/el if self.stelap > 0 else None)

        # Decide how many '+', '=' and ' ' to output for the progress bar
        n = self._nbarcol - len(wela) - len(wrem) - 1
        nps = int(n * (rcu - el)/ren)
        neq = int(n * el/ren)
        nsp = n - nps - neq

        # Render the progress bar
        s = self._dispfmt.format(frac, '+'*nps, '='*neq, ' '*nsp, cu, en,
                                 wela, wrem, dps=self.dps)

        # Write the progress bar and pad the remaining columns
        sys.stderr.write('\x1b[2K\x1b[G')
        sys.stderr.write(s)
        sys.stderr.flush()

        # Update the last render time
        self._last_wallt = wallt
