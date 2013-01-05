# -*- coding: utf-8 -*-

import os
import sys
import time

def get_terminal_size():
    if sys.platform in ('linux2', 'darwin'):
        return map(int, os.popen('stty size', 'r').read().split())
    else:
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
    _dispfmt = '{:7.1%} [{}{}>{}] {:.2f}/{:.2f} ela: {} rem: {}'

    # Minimum time in seconds between updates
    _mindelay = 0.1

    def __init__(self, start, curr, end):
        self.ststrt = start
        self.stelap = 0.0
        self.stcurr = curr
        self.stend = end

        self._wstart = time.time()

        self._ncol = get_terminal_size()[1]
        self._nbarcol = self._ncol - 60

        self._last_render = 0.0
        self._render()

    def advance_to(self, t):
        self.stelap = min(t, self.stend) - self.ststrt
        self.stcurr = min(t, self.stend)

        self._render()

        if self.stcurr == self.stend:
            sys.stderr.write('\n')

    @property
    def walltime(self):
        return time.time() - self._wstart

    def _render(self):
        # If we have rendered recently then do not do so again
        if (self.walltime - self._last_render) < self._mindelay:
            return

        # Curr, elapsed and ending simulation times
        cu, el, en = self.stcurr, self.stelap, self.stend

        # Fraction of the simulation we've completed
        frac = cu / en

        # Decide how many '+', '=' and ' ' to output for the progress bar
        n = self._nbarcol - 1;
        nps = int(n * (cu - el)/en)
        neq = int(n * el/en)
        nsp = n - nps - neq;

        # Elapsed wall time
        wela = format_hms(self.walltime)

        # Remaining wall time
        if self.stelap > 0:
            trem = self.walltime*(en - cu)/el
        else:
            trem = None
        wrem = format_hms(trem)

        # Render the progress bar
        s = self._dispfmt.format(frac, '+'*nps, '='*neq, ' '*nsp, cu, en,
                                 wela, wrem)

        # Write the progress bar and pad the remaining columns
        sys.stderr.write(chr(27) + '[2K' + chr(27) +'[G')
        sys.stderr.write(s)
        sys.stderr.write(' '*(self._ncol - len(s)))
        sys.stderr.flush()

        # Update the last render time
        self._last_render = self.walltime
