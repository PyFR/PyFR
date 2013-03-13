# -*- coding: utf-8 -*-

import sys


class _ExceptHook(object):
    def __init__(self):
        self.exception = None

        self._orig_excepthook = sys.excepthook
        sys.excepthook = self._excepthook

    def _excepthook(self, exc_type, exc, *args):
        self.exception = exc
        self._orig_excepthook(exc_type, exc, *args)


# Global instance
excepthook = _ExceptHook()
